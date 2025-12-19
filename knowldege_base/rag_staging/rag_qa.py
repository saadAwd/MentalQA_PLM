"""
End-to-end RAG QA over the knowledge base (staging).

Flow:
- Use `HybridKBRetriever` to fetch top-K relevant chunks.
- Concatenate them into a context.
- Use a seq2seq generation model (via `transformers`) to answer in Arabic.

Default model: a small multilingual seq2seq model suitable for CPU:
- `google/mt5-small`

You can override the model name via the `RAG_QA_MODEL_NAME` environment
variable or the `--model-name` CLI argument.

Usage (from project root, venv activated):

    python -m knowldege_base.rag_staging.rag_qa "ما هو الاكتئاب؟"
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

from .hybrid_retriever import HybridKBRetriever
from . import loader


DEFAULT_MODEL_NAME = os.environ.get("RAG_QA_MODEL_NAME", "google/mt5-small")

# Local model cache directory
MODELS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "models"
)


def _get_local_model_path(model_name: str, quantized: bool = False) -> str:
    """
    Get local path for a model.
    
    Args:
        model_name: HuggingFace model name (e.g., "Sakalti/Saka-14B")
        quantized: Whether this is a quantized version
    
    Returns:
        Local directory path for the model
    """
    # Sanitize model name for filesystem
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    suffix = "_4bit" if quantized else ""
    return os.path.join(MODELS_CACHE_DIR, f"{safe_name}{suffix}")


def _model_exists_locally(model_name: str, quantized: bool = False) -> bool:
    """Check if model exists in local cache."""
    local_path = _get_local_model_path(model_name, quantized)
    # Check if config.json exists (indicates model is saved)
    config_path = os.path.join(local_path, "config.json")
    return os.path.exists(config_path)


def _download_model_to_local(model_name: str) -> str:
    """
    Download model from HuggingFace to local project directory.
    
    Returns:
        Local path where model is saved
    """
    from huggingface_hub import snapshot_download
    
    local_path = _get_local_model_path(model_name, quantized=False)
    
    # Check if already downloaded
    if _model_exists_locally(model_name, quantized=False):
        print(f"✓ Model already exists locally: {local_path}")
        return local_path
    
    print(f"Downloading model to local directory: {local_path}")
    print("This may take a while for large models...")
    
    # Download to local directory
    os.makedirs(local_path, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Don't use symlinks on Windows
            resume_download=True,
        )
        print(f"✓ Model downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        raise


def _save_model_locally(model, tokenizer, model_name: str, quantized: bool = False):
    """Save model and tokenizer to local cache."""
    local_path = _get_local_model_path(model_name, quantized)
    os.makedirs(local_path, exist_ok=True)
    
    print(f"Saving model to local cache: {local_path}")
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    print(f"✓ Model saved locally")


@dataclass
class RAGAnswer:
    query: str
    answer: str
    used_chunks: List[Dict]
    used_kb: bool = True  # Whether knowledge base was used
    top_score: float = 0.0  # Top retrieval score
    avg_top_score: float = 0.0  # Average of top 3 scores


def _build_context_from_chunks(chunks: List[Dict], max_chars: int = 4000) -> str:
    """
    Concatenate chunk texts into a single context string, truncated to max_chars.
    Filters out obviously corrupted chunks and formats them clearly.
    """
    parts: List[str] = []
    total = 0

    for i, ch in enumerate(chunks, 1):
        title = ch.get("title") or ""
        text = ch.get("text") or ""
        
        # CRITICAL FIX: For QA pairs, extract only the ANSWER part
        # Remove question prefixes that might confuse the generator
        if ch.get("kb_family") == "qa_pair" or ch.get("content_type") == "qa_pair":
            # Remove common question prefixes
            question_prefixes = [
                "السؤال:",
                "السؤال",
                "الاستشارة:",
                "الاستشارة",
            ]
            for prefix in question_prefixes:
                if text.startswith(prefix):
                    # Find where the answer starts (usually after "الإجابة:" or newline)
                    parts = text.split("الإجابة:", 1)
                    if len(parts) > 1:
                        text = parts[1].strip()
                    else:
                        # If no "الإجابة:" marker, remove the question prefix and take rest
                        text = text[len(prefix):].strip()
                    break
        
        # Skip chunks with obviously corrupted text
        if text:
            # Count single character words (likely OCR corruption)
            words = text.split()
            single_char_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
            if len(words) > 10 and single_char_words > len(words) * 0.4:
                print(f"⚠ Skipping chunk {i} due to heavy corruption")
                continue
        
        # Don't include title for QA pairs (title is the question, we don't want it)
        header = f"[مصدر {i}]"
        if title and ch.get("kb_family") != "qa_pair":
            header += f" {title}"
        header += "\n"
        
        block = header + text.strip()
        if not block or len(text.strip()) < 20:  # Skip very short chunks
            continue

        # Add a separator between chunks
        block = block.strip()
        to_add = "\n\n---\n\n" + block if parts else block

        if total + len(to_add) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            to_add = to_add[:remaining]
            parts.append(to_add)
            total += len(to_add)
            break

        parts.append(to_add)
        total += len(to_add)

    return "\n".join(parts).strip()


def _build_prompt_ar(query: str, context: str, tokenizer=None, use_chat_template: bool = False) -> str:
    """
    Build a professional medical prompt using user-provided instructions.
    Supports two versions: 'original' (no filtering) and 'filtering' (explicit instructions).
    """
    import os
    
    # Truncate context if too long
    max_context_length = 3000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    # Get prompt version from environment variable (default: 'filtering')
    prompt_version = os.environ.get('RAG_PROMPT_VERSION', 'filtering').lower()
    
    # Version A: Original prompt (no filtering instructions)
    if prompt_version == 'original':
        system_instruction = (
            "أنت طبيب / معالج نفسي افتراضي محترف.\n\n"
            "مهمتك أن ترد على السؤال أو طلب الاستشارة النفسية باللغة العربية الفصحى الواضحة.\n\n"
            "إرشادات الإجابة:\n"
            "- قدّم تفسيرا نفسيا مبسطا ومهنيّا قدر الإمكان.\n"
            "- إذا كان هناك أي احتمال لخطورة (مثل إيذاء النفس أو الآخرين)، شجّع السائل بقوة على طلب المساعدة الفورية من مختص أو جهة طوارئ في بلده.\n"
            "- لا تذكر أسماء أدوية ولا تقدّم تشخيصا طبيا قاطعا، بل ركّز على التوجيه والدعم العام.\n"
            "- استعن بالمعلومات الواردة في 'السياق المتاح' لتوفير إجابة دقيقة ومدعومة بالحقائق."
        )
        user_instruction = (
            f"الاستشارة: {query}\n\n"
            f"السياق المتاح (معلومات طبية للاستعانة بها):\n{context}\n\n"
            f"قم بكتابة الرد الطبي على هذه الاستشارة باللغة العربية الفصحى."
        )
    else:
        # Version B: Filtering prompt (explicit instructions to avoid Quranic content)
        system_instruction = (
            "أنت طبيب / معالج نفسي افتراضي محترف.\n\n"
            "مهمتك أن ترد على السؤال أو طلب الاستشارة النفسية باللغة العربية الفصحى الواضحة.\n\n"
            "إرشادات الإجابة:\n"
            "- قدّم تفسيرا نفسيا مبسطا ومهنيّا قدر الإمكان.\n"
            "- إذا كان هناك أي احتمال لخطورة (مثل إيذاء النفس أو الآخرين)، شجّع السائل بقوة على طلب المساعدة الفورية من مختص أو جهة طوارئ في بلده.\n"
            "- لا تذكر أسماء أدوية ولا تقدّم تشخيصا طبيا قاطعا، بل ركّز على التوجيه والدعم العام.\n"
            "- استعن بالمعلومات الواردة في 'السياق المتاح' لتوفير إجابة دقيقة ومدعومة بالحقائق.\n"
            "- **مهم جداً**: استخدم اللغة العربية فقط في إجابتك. لا تستخدم أي نصوص قرآنية أو آيات قرآنية.\n"
            "- **مهم جداً**: إذا كان السياق يحتوي على نصوص دينية أو قرآنية، تجاهلها وركز على المعلومات الطبية والنفسية فقط."
        )
        user_instruction = (
            f"الاستشارة: {query}\n\n"
            f"السياق المتاح (معلومات طبية للاستعانة بها):\n{context}\n\n"
            f"قم بكتابة الرد الطبي على هذه الاستشارة:\n"
            f"- استخدم اللغة العربية فقط\n"
            f"- لا تنسخ أو تستشهد بأي نصوص قرآنية أو آيات\n"
            f"- ركز على المعلومات الطبية والنفسية من السياق فقط"
        )

    # Try chat template for Qwen/Saka models
    if use_chat_template and tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt
        except Exception as e:
            print(f"⚠ Chat template failed ({e}), using simple format")
    
    # Simple format for MT5 and other models
    prompt = (
        f"{system_instruction}\n\n"
        f"{user_instruction}"
    )
    return prompt


def _extract_fallback_answer(query: str, chunks: List[Dict]) -> str:
    """
    Fallback: extract the most relevant sentence from chunks if generation fails.
    Simple heuristic: find chunk with query keywords and return first few sentences.
    """
    query_words = set(query.lower().split())
    
    # Find best chunk by keyword overlap
    best_chunk = None
    best_score = 0
    
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        score = sum(1 for word in query_words if word in text)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    if best_chunk:
        text = best_chunk.get("text", "")
        # Extract first 2-3 sentences
        sentences = text.split(".")[:3]
        answer = ". ".join(s.strip() for s in sentences if s.strip())
        if answer:
            return answer + "."
    
    # Last resort: return first chunk's first paragraph
    if chunks:
        text = chunks[0].get("text", "")
        if text:
            # Get first ~200 chars
            answer = text[:200].strip()
            if answer:
                return answer + "..."
    
    return "لم أتمكن من العثور على إجابة كافية في قاعدة المعرفة المتاحة."


class RAGQAPipeline:
    """
    Simple RAG QA pipeline:
    - Hybrid retrieval over KB chunks.
    - Seq2seq generation conditioned on retrieved context.
    """

    def __init__(
        self,
        retriever: HybridKBRetriever,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 256,
        use_gpu: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        download_to_local: bool = False,
    ) -> None:
        self.retriever = retriever
        self.original_model_name = model_name  # Store original name for saving
        self.model_name = model_name
        self.save_quantized = True  # Default to saving quantized models
        self._using_device_map_auto = False  # Track if using device_map="auto"
        
        # Check if model is a local path or should be loaded from local directory
        # If download_to_local is True, use local path (don't download, just use if exists)
        if download_to_local:
            local_path = _get_local_model_path(model_name, quantized=False)
            if _model_exists_locally(model_name, quantized=False):
                print(f"✓ Using locally cached model: {local_path}")
                self.model_name = local_path  # Use local path instead of HuggingFace name
            else:
                print(f"⚠ Local model not found at: {local_path}")
                print(f"  Model will be loaded from HuggingFace cache (if available)")
                print(f"  To download to local directory, use: python -m knowldege_base.rag_staging.download_model {model_name}")
        
        # Also check if model_name is already a local path
        if os.path.exists(model_name) and os.path.isdir(model_name):
            print(f"✓ Using local model path: {model_name}")
            self.model_name = model_name

        # Detect device
        cuda_available = torch.cuda.is_available()
        if use_gpu and cuda_available:
            device = "cuda"
            device_id = 0
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.2f} GB)")
            if gpu_memory < 10 and (load_in_4bit or load_in_8bit):
                print(f"⚠ GPU has {gpu_memory:.2f}GB - quantization is essential for this model size")
        else:
            device = "cpu"
            device_id = -1
            if use_gpu and not cuda_available:
                print("⚠ GPU requested but CUDA not available!")
                print("  Check: 1) PyTorch CUDA installation 2) NVIDIA drivers 3) GPU hardware")
                print("  For CPU, a 14B model will be VERY slow (may take 10+ minutes per query)")
            else:
                print("Using CPU")
            load_in_4bit = False  # Quantization only works on GPU
            load_in_8bit = False
        
        # Store device for embedding model
        self.device = device

        # Check if model is seq2seq (like MT5) or causal (like Saka-14B)
        is_causal = "saka" in model_name.lower() or "gpt" in model_name.lower() or "llama" in model_name.lower() or "qwen" in model_name.lower()
        
        # Check if model is MLX format (Apple Silicon only - not compatible with PyTorch)
        is_mlx = "mlx-community" in model_name.lower() or "/mlx" in model_name.lower()
        if is_mlx:
            raise ValueError(
                f"Model {model_name} is in MLX format (Apple Silicon only). "
                f"For NVIDIA GPU, use the original model 'Sakalti/Saka-14B' with --load-in-4bit flag."
            )
        
        # Check if model is already quantized in PyTorch format (not MLX)
        is_pre_quantized = (
            ("4bit" in model_name.lower() or "4-bit" in model_name.lower() or 
             "8bit" in model_name.lower() or "8-bit" in model_name.lower() or 
             "gguf" in model_name.lower())
            and not is_mlx
        )
        
        # Determine if we need quantization
        needs_quantization = (load_in_4bit or load_in_8bit) and device == "cuda" and not is_pre_quantized
        will_quantize = needs_quantization
        
        # Check for local cached quantized model
        use_local = False
        local_path = None
        original_model_name_for_save = model_name  # Keep original for saving later
        if will_quantize:
            local_path = _get_local_model_path(original_model_name_for_save, quantized=True)
            if _model_exists_locally(original_model_name_for_save, quantized=True):
                print(f"✓ Found locally cached quantized model: {local_path}")
                print("  Loading from local cache (no download/quantization needed)")
                use_local = True
                model_name = local_path  # Use local path instead
                will_quantize = False  # Already quantized
                # Still need quantization config for loading (model was saved quantized)
                if load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        print(f"Loading QA generation model: {model_name}")
        # Check if it's a local path
        if os.path.exists(model_name) and os.path.isdir(model_name):
            print(f"  Loading from local directory (no download)")
        else:
            print(f"  Loading from HuggingFace (will use cache if available)")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
        
        # Check if model is seq2seq (like MT5) or causal (like Saka-14B)
        is_causal = "saka" in model_name.lower() or "gpt" in model_name.lower() or "llama" in model_name.lower() or "qwen" in model_name.lower()
        if not is_causal:
            is_causal = tokenizer.is_fast and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        
        # Check if tokenizer has chat_template (for Qwen/Saka models)
        has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        self.has_chat_template = has_chat_template
        self.is_causal = is_causal
        
        # Setup quantization for large models (only if NOT pre-quantized and NOT using local)
        quantization_config = None
        if will_quantize:
            if load_in_4bit:
                print("Using 4-bit quantization to fit in GPU memory (~7GB)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                print("Using 8-bit quantization to fit in GPU memory (~14GB)")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif is_pre_quantized:
            print("Model appears to be pre-quantized, skipping quantization setup")
        elif is_causal and device == "cuda" and not load_in_4bit and not load_in_8bit:
            # Warn if large model without quantization
            print("⚠ Warning: Large causal model detected. Consider using --load-in-4bit for 8.55GB GPU")

        # Load model based on type
        if is_causal:
            # Causal LM (like Saka-14B, GPT, etc.)
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
                    self._using_device_map_auto = True
                else:
                    model_kwargs["device_map"] = None
            
            try:
                # Only try to download if not a local path
                load_kwargs = model_kwargs.copy()
                if os.path.exists(model_name) and os.path.isdir(model_name):
                    # Local path - don't try to download
                    load_kwargs["local_files_only"] = True
                else:
                    # HuggingFace model - allow download but prefer cache
                    load_kwargs["local_files_only"] = False
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs,
                )
            except RuntimeError as e:
                error_str = str(e)
                # Check if it's a bitsandbytes CUDA kernel error (sm_120 not supported)
                if "no kernel image is available" in error_str or ("CUDA error" in error_str and "bitsandbytes" in error_str.lower()):
                    print(f"\n⚠ bitsandbytes quantization failed: {error_str[:200]}")
                    print("  This is because RTX 5060 (sm_120) is not yet supported by bitsandbytes")
                    print("\n  Trying fallback: Loading without quantization, using CPU offloading...")
                    
                    # Fallback: Try without quantization, use CPU offloading
                    if quantization_config:
                        model_kwargs_fallback = {
                            "trust_remote_code": True,
                            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                            "device_map": "auto",  # Automatically offload to CPU/GPU
                            "low_cpu_mem_usage": True,
                        }
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                **model_kwargs_fallback,
                                resume_download=True,
                            )
                            print("  ✓ Model loaded with CPU offloading (slower but works)")
                            print("  ⚠ Note: Model will use both GPU and CPU memory")
                            quantization_config = None  # Clear quantization config
                            will_quantize = False  # Don't try to save as quantized
                            # Mark that we're using device_map="auto" so pipeline doesn't set device
                            self._using_device_map_auto = True
                        except Exception as e2:
                            print(f"  ✗ CPU offloading also failed: {e2}")
                            print("\n  💡 Alternative solutions:")
                            print("    1. Wait for bitsandbytes to support sm_120")
                            print("    2. Use a smaller model that fits without quantization")
                            print("    3. Use a pre-quantized model (GPTQ/AWQ format)")
                            raise RuntimeError(
                                "Model loading failed. bitsandbytes doesn't support RTX 5060 (sm_120) yet.\n"
                                "Try: 1) Use a smaller model, 2) Wait for bitsandbytes update, 3) Use pre-quantized model"
                            ) from e2
                    else:
                        raise e
                else:
                    raise e
            except Exception as e:
                error_str = str(e)
                # Check if it's a download/connection error
                if "IncompleteRead" in error_str or "Connection" in error_str or "timeout" in error_str.lower() or "401" in error_str or "404" in error_str:
                    print(f"\n✗ Model loading error: {error_str[:200]}")
                    print("\n💡 Solutions:")
                    print(f"   1. Download model to local directory first:")
                    print(f"      python -m knowldege_base.rag_staging.download_model {self.original_model_name}")
                    print(f"   2. Then use --download-to-local flag to load from local path")
                    print(f"   3. Or use huggingface-cli:")
                    print(f"      huggingface-cli download {self.original_model_name}")
                    raise RuntimeError(
                        f"Model not found locally and download failed.\n"
                        f"Download manually first:\n"
                        f"  python -m knowldege_base.rag_staging.download_model {self.original_model_name}\n"
                        f"Then use --download-to-local flag."
                    ) from e
                raise
            
            # Save quantized model locally if we just quantized it
            if will_quantize and quantization_config and not use_local and self.save_quantized:
                _save_model_locally(model, tokenizer, original_model_name_for_save, quantized=True)
            
            # Check if model is using device_map="auto" (CPU offloading)
            using_device_map_auto = self._using_device_map_auto
            # Also check if model was loaded with device_map (from accelerate)
            if not using_device_map_auto:
                # Check if model has device_map in its config (from accelerate)
                try:
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        using_device_map_auto = True
                        self._using_device_map_auto = True
                except:
                    pass
            
            if device == "cpu" and not using_device_map_auto:
                model = model.to(device)
            
            # Use text-generation pipeline for causal models
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
            }
            
            # Don't set device if using device_map="auto" (managed by accelerate)
            if not using_device_map_auto:
                pipeline_kwargs["device"] = device_id
                pipeline_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            else:
                # When using device_map="auto", let accelerate handle device placement
                pipeline_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            
            self.generator = pipeline("text-generation", **pipeline_kwargs)
            self.is_causal = True
        else:
            # Seq2seq model (like MT5, MBART)
            # Check if local path
            if os.path.exists(model_name) and os.path.isdir(model_name):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=False)
            if device == "cuda":
                model = model.to(device)
            
            self.generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id,
            )
            self.is_causal = False
        
        self.max_new_tokens = max_new_tokens
        self.device = device

    @classmethod
    def build(
        cls,
        model_name: Optional[str] = None,
        alpha: Optional[float] = None,
        max_new_tokens: int = 256,
        use_gpu: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        save_quantized: bool = True,
        download_to_local: bool = False,
    ) -> "RAGQAPipeline":
        try:
            from rag_config import RAGConfig
            if alpha is None:
                alpha = RAGConfig.HYBRID_ALPHA
        except ImportError:
            if alpha is None:
                alpha = 0.7
                
        # Build retriever strictly on CPU to maximize VRAM for the 7B generator
        retriever = HybridKBRetriever.build(alpha=alpha, use_cpu=True)
        pipeline = cls(
            retriever=retriever,
            model_name=model_name or DEFAULT_MODEL_NAME,
            max_new_tokens=max_new_tokens,
            use_gpu=use_gpu,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            download_to_local=download_to_local,
        )
        pipeline.save_quantized = save_quantized
        return pipeline

    def answer(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 4000,
        relevance_threshold: float = 0.5,
        rerank: bool = True,
    ) -> RAGAnswer:
        """
        Generate answer with relevance threshold check.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            max_context_chars: Maximum context length
            relevance_threshold: Minimum score (0-1) to use KB. 
            rerank: Whether to use re-ranking if available
        """
        # 1) Retrieve top-K chunks (with re-ranking if enabled)
        results = self.retriever.search(query, top_k=top_k, rerank=rerank)
        
        # Check relevance threshold
        # If we have re-ranked scores, they are usually in results[0]["score_hybrid"]
        # which now stores the re-ranked score.
        
        # We'll use the average of the top 3 scores for the threshold check
        if results:
            # Use the score_hybrid which is now the re-ranked score if rerank was True
            top_scores = [r["score_hybrid"] for r in results[:3]]
            avg_top_score = sum(top_scores) / len(top_scores)
            top_score = results[0]["score_hybrid"]
            
            # Use raw scores for metadata/display
            top_raw_score = results[0].get("score_raw_dense", 0.0)
            avg_top_raw_score = sum(r.get("score_raw_dense", 0.0) for r in results[:3]) / len(results[:3])
            
            # Decision: Use KB if avg_top_score >= relevance_threshold
            # Note: Cross-encoder scores for multilingual-MiniLM-L12-v2 are usually sigmoid-based (0 to 1)
            use_kb = avg_top_score >= relevance_threshold
        else:
            top_score = 0.0
            avg_top_score = 0.0
            top_raw_score = 0.0
            avg_top_raw_score = 0.0
            use_kb = False
        
        # Debug: Show scores
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 3:
            rerank_status = " (RERANKED)" if rerank and self.retriever.rerank_model else ""
            print(f"[DEBUG] Query {self._debug_count + 1}{rerank_status} - Avg top score: {avg_top_score:.4f}, Threshold: {relevance_threshold:.4f}")
            self._debug_count += 1
        
        if not use_kb:
            if results:
                print(f"⚠ Top score (avg: {avg_top_score:.3f}) below threshold ({relevance_threshold:.3f})")
            else:
                print(f"⚠ No retrieval results found")
            print("  Generating answer without knowledge base context...")
            
            # Generate without KB context
            if self.is_causal:
                prompt_no_kb = f"السؤال: {query}\n\nالإجابة:"
            else:
                prompt_no_kb = f"السؤال: {query}\n\nالإجابة:"
            
            try:
                if self.is_causal:
                    # Causal LM generation
                    gen_outputs = self.generator(
                        prompt_no_kb,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.generator.tokenizer.eos_token_id,
                    )
                    # Extract only the generated part (remove input prompt)
                    full_text = gen_outputs[0]["generated_text"]
                    answer_text = full_text[len(prompt_no_kb):].strip()
                else:
                    # Seq2seq generation
                    gen_outputs = self.generator(
                        prompt_no_kb,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    answer_text = gen_outputs[0]["generated_text"].strip()
                
                answer_text = answer_text.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
                answer_text = answer_text.strip()
                
                # If generation fails or is empty, provide a helpful message
                if not answer_text or len(answer_text) < 10:
                    answer_text = "عذراً، لا أستطيع الإجابة على هذا السؤال بناءً على قاعدة المعرفة المتاحة. يرجى طرح سؤال متعلق بالصحة النفسية."
            except Exception as e:
                print(f"⚠ Generation failed ({e})")
                answer_text = "عذراً، لا أستطيع الإجابة على هذا السؤال بناءً على قاعدة المعرفة المتاحة. يرجى طرح سؤال متعلق بالصحة النفسية."
            
            return RAGAnswer(
                query=query,
                answer=answer_text,
                used_chunks=[],
                used_kb=False,
                top_score=top_score,
                avg_top_score=avg_top_score if results else 0.0,
            )
        
        # 2) Build context from retrieved chunks
        chunks = [r["chunk"] for r in results]
        context = _build_context_from_chunks(chunks, max_chars=max_context_chars)
        prompt = _build_prompt_ar(query, context, tokenizer=self.generator.tokenizer, use_chat_template=self.has_chat_template)

        # 3) Generate answer with KB context
        if self.device == "cpu":
            print(f"\n🔄 Generating answer (this may take 3-8 minutes on CPU for 7B model)...")
        else:
            print(f"\n🔄 Generating answer on GPU (should be fast)...")
        print(f"   Using {len(chunks)} chunks from knowledge base")
        try:
            if self.is_causal:
                # Causal LM generation - use model directly for better control
                print("   Generating with causal LM...")
                
                # Access underlying model and tokenizer from pipeline
                model = self.generator.model
                tokenizer = self.generator.tokenizer
                
                # Tokenize prompt
                prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
                # CRITICAL: Always move tokens to the same device as the model
                # Even with device_map="auto", input_ids must be on the correct device
                if self.device != "cpu":
                    # Get the device of the model's first parameter
                    model_device = next(model.parameters()).device
                    prompt_tokens = prompt_tokens.to(model_device)
                else:
                    prompt_tokens = prompt_tokens.to(self.device)
                input_length = prompt_tokens.shape[1]
                
                # Use pipeline's generate method (more reliable - handles formatting correctly)
                # The pipeline method is simpler and handles all edge cases
                gen_outputs = self.generator(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
                
                # Extract answer from pipeline output
                full_text = gen_outputs[0]["generated_text"]
                
                # Extract only the new part (remove prompt)
                if prompt in full_text:
                    answer_text = full_text[len(prompt):].strip()
                else:
                    # If prompt not found, try to find where answer starts
                    # Look for common Arabic answer markers
                    answer_markers = ["الإجابة:", "الرد:", "الجواب:", "بناءً على", "الاستشارة:"]
                    answer_text = full_text
                    for marker in answer_markers:
                        if marker in full_text:
                            parts = full_text.split(marker, 1)
                            if len(parts) > 1:
                                answer_text = parts[1].strip()
                                break
                    
                    # If still no good extraction, use everything after first 100 chars (skip prompt)
                    if len(answer_text) == len(full_text) and len(full_text) > 100:
                        answer_text = full_text[100:].strip()
                
                # If answer is mostly punctuation, something went wrong - use extractive fallback
                if len(answer_text) > 0:
                    non_punct = sum(1 for c in answer_text if c.isalnum() or c.isspace())
                    if non_punct < len(answer_text) * 0.1:  # Less than 10% non-punctuation
                        print(f"   [DEBUG] Answer is mostly punctuation ({len(answer_text)} chars, {non_punct} non-punct)")
                        print(f"   [DEBUG] Full text preview: {full_text[:500]}...")
                        print(f"   [DEBUG] Using extractive fallback...")
                        answer_text = _extract_fallback_answer(query, chunks)
                
                # Additional cleanup for garbled text detection
                # Check if output looks like garbage (too many single characters, repeated patterns)
                if len(answer_text) > 20:
                    # Count single character words (likely garbage)
                    words = answer_text.split()
                    single_char_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
                    if len(words) > 5 and single_char_words > len(words) * 0.4:  # More than 40% single char words
                        print("⚠ Generated text appears garbled, using extractive fallback...")
                        answer_text = _extract_fallback_answer(query, chunks)
                    # Also check if answer is too similar to source chunks (likely copying)
                    elif any(chunk.get("text", "")[:100] in answer_text[:200] for chunk in chunks):
                        print("⚠ Answer appears to be copied from source, trying to improve...")
                        # Try to extract a better answer
                        answer_text = _extract_fallback_answer(query, chunks)
                
                print("✓ Answer generated successfully")
            else:
                # Seq2seq generation (MT5, etc.)
                gen_outputs = self.generator(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                answer_text = gen_outputs[0]["generated_text"].strip()
            
            # Clean up any special tokens that might appear
            answer_text = answer_text.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
            answer_text = answer_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
            answer_text = answer_text.replace("<|endoftext|>", "")
            
            # Remove mixed-language content (non-Arabic/Latin characters)
            import re
            # Remove Chinese, Japanese, Korean characters (CJK Unified Ideographs)
            # Keep Arabic, Latin, numbers, punctuation, and common symbols
            # Remove: \u4E00-\u9FFF (CJK), \u3400-\u4DBF (CJK Extension A), etc.
            answer_text = re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF]', '', answer_text)
            
            answer_text = answer_text.strip()
            
            # Debug: Print first 200 chars of generated text to diagnose issues
            if len(answer_text) < 50:
                print(f"   [DEBUG] Generated text (first 200 chars): {full_text[:200]}")
                print(f"   [DEBUG] Extracted answer length: {len(answer_text)}")
            
            # Check if generation actually produced meaningful content
            # Relaxed threshold: minimum 20 characters (was 10) to allow for short but valid answers
            # Also check if it's just special tokens or punctuation
            is_just_punctuation = answer_text.strip() in [".", "،", "؟", "!", ":", ";", ","]
            is_too_short = len(answer_text) < 20
            is_only_special_tokens = len(answer_text.replace(" ", "").replace("\n", "")) < 10
            
            if not answer_text or is_too_short or is_just_punctuation or is_only_special_tokens:
                print(f"⚠ Generation produced empty/short output (len={len(answer_text)}), using extractive fallback...")
                answer_text = _extract_fallback_answer(query, chunks)
        except Exception as e:
            print(f"⚠ Generation failed ({e}), using extractive fallback...")
            answer_text = _extract_fallback_answer(query, chunks)

        return RAGAnswer(
            query=query,
            answer=answer_text,
            used_chunks=chunks,
            used_kb=True,
            top_score=top_score,
            avg_top_score=avg_top_score if results else 0.0,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end RAG QA over knowledge base (Arabic)."
    )
    parser.add_argument("query", type=str, nargs="?", help="Arabic user question.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF model name for generation (defaults to env RAG_QA_MODEL_NAME or google/mt5-small).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,  # Reduced default for faster CPU generation
        help="Maximum new tokens to generate (default: 128 for CPU, use 256 for GPU).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (defaults to rag_qa_results_utf8.txt in processed data dir).",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.65,
        help="Minimum retrieval score (0-1) to use KB. Default: 0.65 (65%%) - increased for better relevance",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        help="Force CPU usage",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization for GPU memory efficiency (default: True for large models)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (alternative to 4-bit)",
    )
    parser.add_argument(
        "--no-save-quantized",
        action="store_false",
        dest="save_quantized",
        default=True,
        help="Don't save quantized model to local cache",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage (useful for RTX 5060 sm_120 compatibility issues)",
    )
    parser.add_argument(
        "--download-to-local",
        action="store_true",
        help="Download model to local project directory instead of HuggingFace cache",
    )

    args = parser.parse_args()

    if not args.query:
        # Simple default Arabic test question
        query = "ما هو الاكتئاب؟"
        print(
            "No query provided on the command line. "
            "Using default test query (Arabic: 'ما هو الاكتئاب؟')"
        )
    else:
        query = args.query

    # If --use-cpu is set, override use_gpu
    use_gpu = args.use_gpu and not args.use_cpu
    
    rag = RAGQAPipeline.build(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        use_gpu=use_gpu,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        save_quantized=args.save_quantized,
        download_to_local=args.download_to_local,
    )

    result = rag.answer(
        query=query,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )

    # Determine output file path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "processed",
            "rag_qa_results_utf8.txt"
        )
        output_path = os.path.abspath(output_path)

    # Write results to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAG QA Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"السؤال: {result.query}\n")
        f.write("=" * 80 + "\n\n")
        f.write("الإجابة:\n")
        f.write(result.answer)
        f.write("\n\n" + "=" * 80 + "\n\n")
        f.write(f"استخدام قاعدة المعرفة: {'نعم' if result.used_kb else 'لا'}\n")
        f.write(f"أعلى درجة استرجاع: {result.top_score:.3f}\n")
        if hasattr(result, 'avg_top_score'):
            f.write(f"متوسط أعلى 3 درجات: {result.avg_top_score:.3f}\n")
        f.write(f"عتبة الصلة: {args.relevance_threshold:.3f}\n")
        f.write(f"عدد المقاطع المستخدمة: {len(result.used_chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Include chunk details for reference (only if KB was used)
        if result.used_chunks:
            f.write("المقاطع المستخدمة:\n")
            f.write("-" * 80 + "\n")
            for idx, chunk in enumerate(result.used_chunks, 1):
                f.write(f"\n[{idx}] Chunk ID: {chunk.get('chunk_id', 'N/A')}\n")
                f.write(f"Title: {chunk.get('title', 'N/A')}\n")
                f.write(f"Source: {chunk.get('content_type', 'N/A')}\n")
                f.write(f"URL: {chunk.get('url', 'N/A')}\n")
                f.write(f"\nText Preview:\n{chunk.get('text', '')[:300]}...\n")
                f.write("\n" + "-" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"السؤال: {result.query}")
    print("=" * 80)
    print("الإجابة:\n")
    print(result.answer)
    print("\n" + "=" * 80)
    print(f"استخدام قاعدة المعرفة: {'نعم' if result.used_kb else 'لا'}")
    print(f"أعلى درجة استرجاع: {result.top_score:.3f}")
    print(f"عتبة الصلة: {args.relevance_threshold:.3f}")
    print(f"عدد المقاطع المستخدمة: {len(result.used_chunks)}")
    print("=" * 80)
    print(f"\n✓ Results written to: {output_path}")
    print("Note: Full results with chunk details are in the file above (UTF-8 encoding).\n")


if __name__ == "__main__":
    main()


