"""Sophisticated data augmentation for multi-label Arabic text classification"""
import random
import json
from typing import List, Dict, Optional
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class ArabicDataAugmenter:
    """Advanced Arabic text augmentation using back-translation and paraphrasing"""
    
    def __init__(self, seed: int = 42, device: Optional[str] = None):
        random.seed(seed)
        self.seed = seed
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize translation models (lazy loading)
        self._en_ar_translator = None
        self._ar_en_translator = None
        self._paraphrase_model = None
        self._paraphrase_tokenizer = None
        self._qwen_model = None
        self._qwen_tokenizer = None
    
    def _get_en_ar_translator(self):
        """Get English to Arabic translator (lazy loading)"""
        if self._en_ar_translator is None:
            try:
                logger.info("Loading English->Arabic translation model...")
                self._en_ar_translator = pipeline(
                    "translation_en_to_ar",
                    model="Helsinki-NLP/opus-mt-en-ar",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("English->Arabic translator loaded")
            except Exception as e:
                logger.warning(f"Could not load English->Arabic translator: {e}")
                self._en_ar_translator = False  # Mark as unavailable
        return self._en_ar_translator if self._en_ar_translator is not False else None
    
    def _get_ar_en_translator(self):
        """Get Arabic to English translator (lazy loading)"""
        if self._ar_en_translator is None:
            try:
                logger.info("Loading Arabic->English translation model...")
                self._ar_en_translator = pipeline(
                    "translation_ar_to_en",
                    model="Helsinki-NLP/opus-mt-ar-en",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Arabic->English translator loaded")
            except Exception as e:
                logger.warning(f"Could not load Arabic->English translator: {e}")
                self._ar_en_translator = False  # Mark as unavailable
        return self._ar_en_translator if self._ar_en_translator is not False else None
    
    def _get_paraphrase_model(self):
        """Get Arabic paraphrasing model (lazy loading)"""
        if self._paraphrase_model is None:
            try:
                logger.info("Loading Arabic paraphrasing model...")
                # Try to use Arabic T5 or similar model for paraphrasing
                # Fallback to translation-based paraphrasing if not available
                model_name = "UBC-NLP/MARBERT"  # Can use for embedding-based similarity
                # For now, we'll use back-translation as paraphrasing
                self._paraphrase_model = True  # Mark as available (using back-translation)
                logger.info("Paraphrasing available (using back-translation)")
            except Exception as e:
                logger.warning(f"Could not load paraphrasing model: {e}")
                self._paraphrase_model = False
        return self._paraphrase_model if self._paraphrase_model is not False else None
    
    def back_translate(self, text: str, max_length: int = 512) -> Optional[str]:
        """
        Back-translate: Arabic -> English -> Arabic
        This creates paraphrased versions while preserving meaning
        """
        ar_en = self._get_ar_en_translator()
        en_ar = self._get_en_ar_translator()
        
        if not ar_en or not en_ar:
            return None
        
        try:
            # Step 1: Arabic -> English
            en_text = ar_en(text, max_length=max_length, clean_up_tokenization_spaces=True)
            if isinstance(en_text, list) and len(en_text) > 0:
                en_text = en_text[0].get('translation_text', '')
            elif isinstance(en_text, dict):
                en_text = en_text.get('translation_text', '')
            else:
                en_text = str(en_text)
            
            if not en_text or len(en_text.strip()) < 3:
                return None
            
            # Step 2: English -> Arabic
            ar_text = en_ar(en_text, max_length=max_length, clean_up_tokenization_spaces=True)
            if isinstance(ar_text, list) and len(ar_text) > 0:
                ar_text = ar_text[0].get('translation_text', '')
            elif isinstance(ar_text, dict):
                ar_text = ar_text.get('translation_text', '')
            else:
                ar_text = str(ar_text)
            
            if not ar_text or len(ar_text.strip()) < 3:
                return None
            
            return ar_text.strip()
        except Exception as e:
            logger.debug(f"Back-translation failed: {e}")
            return None
    
    def paraphrase_via_backtranslation(self, text: str) -> Optional[str]:
        """
        Paraphrase using back-translation (same as back_translate but with different name)
        """
        return self.back_translate(text)
    
    def paraphrase_arabic(self, text: str, num_variations: int = 1) -> List[str]:
        """
        Generate Arabic paraphrases using multiple back-translations
        Each back-translation creates a different paraphrase
        """
        paraphrases = []
        for _ in range(num_variations):
            para = self.back_translate(text)
            if para and para not in paraphrases and para != text:
                paraphrases.append(para)
        return paraphrases
    
    def _get_qwen_model(self):
        """Get Qwen model for prompt-based paraphrasing (lazy loading)"""
        if self._qwen_model is None:
            try:
                logger.info("Loading Qwen model for paraphrasing...")
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small, fast model
                # Try Arabic-capable Qwen model
                try:
                    self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self._qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                    if self.device == "cpu":
                        self._qwen_model = self._qwen_model.to(self.device)
                    logger.info("Qwen model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load {model_name}, trying alternative: {e}")
                    # Fallback to a different Qwen model
                    model_name = "Qwen/Qwen2-0.5B-Instruct"
                    self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self._qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                    if self.device == "cpu":
                        self._qwen_model = self._qwen_model.to(self.device)
                    logger.info("Qwen model loaded (fallback)")
            except Exception as e:
                logger.warning(f"Could not load Qwen model: {e}")
                self._qwen_model = False
                self._qwen_tokenizer = False
        return self._qwen_model if self._qwen_model is not False else None, self._qwen_tokenizer if self._qwen_tokenizer is not False else None
    
    def paraphrase_with_qwen(self, text: str, num_paraphrases: int = 10, max_paraphrases: int = 20) -> List[str]:
        """
        Generate Arabic paraphrases using Qwen model with prompt instructions
        
        Args:
            text: Original Arabic text
            num_paraphrases: Number of paraphrases to generate
            max_paraphrases: Maximum number of paraphrases (default 20)
        
        Returns:
            List of paraphrased Arabic texts
        """
        num_paraphrases = min(num_paraphrases, max_paraphrases)
        
        model, tokenizer = self._get_qwen_model()
        if not model or not tokenizer:
            logger.warning("Qwen model not available, falling back to back-translation")
            return self.augment_with_paraphrasing(text, num_augmentations=num_paraphrases)
        
        # Create prompt for paraphrasing
        prompt = f"""أنت مساعد متخصص في إعادة صياغة النصوص العربية. مهمتك هي إنشاء {num_paraphrases} إعادة صياغة مختلفة للنص التالي مع الحفاظ على المعنى الأصلي.

النص الأصلي:
{text}

المطلوب: أنشئ {num_paraphrases} إعادة صياغة مختلفة، كل واحدة في سطر منفصل. استخدم كلمات وعبارات مختلفة مع الحفاظ على المعنى.

الإعادة الصياغة:"""
        
        try:
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with diverse sampling
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract paraphrases from the generated text
            # Remove the prompt part
            if "الإعادة الصياغة:" in generated_text:
                paraphrases_text = generated_text.split("الإعادة الصياغة:")[-1].strip()
            else:
                paraphrases_text = generated_text[len(prompt):].strip()
            
            # Split by lines and clean
            paraphrases = []
            seen = set()
            seen.add(text.lower().strip())
            
            for line in paraphrases_text.split('\n'):
                para = line.strip()
                if para and len(para) >= 5:
                    para_normalized = para.lower().strip()
                    if para_normalized not in seen and para != text:
                        paraphrases.append(para)
                        seen.add(para_normalized)
                        if len(paraphrases) >= num_paraphrases:
                            break
            
            # If we don't have enough, generate more with different prompts
            if len(paraphrases) < num_paraphrases:
                remaining = num_paraphrases - len(paraphrases)
                for _ in range(min(remaining * 2, 10)):  # Try up to 10 more times
                    if len(paraphrases) >= num_paraphrases:
                        break
                    
                    # Alternative prompt
                    alt_prompt = f"""أعد صياغة النص التالي بطريقة مختلفة مع الحفاظ على المعنى:

{text}

الإعادة الصياغة:"""
                    
                    inputs = tokenizer(alt_prompt, return_tensors="pt", truncation=True, max_length=512)
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.9,
                            top_p=0.95,
                            do_sample=True,
                            repetition_penalty=1.3,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    para = generated.split("الإعادة الصياغة:")[-1].strip() if "الإعادة الصياغة:" in generated else generated[len(alt_prompt):].strip()
                    
                    if para and len(para) >= 5:
                        para_normalized = para.lower().strip()
                        if para_normalized not in seen and para != text:
                            paraphrases.append(para)
                            seen.add(para_normalized)
            
            return paraphrases[:num_paraphrases] if paraphrases else []
            
        except Exception as e:
            logger.error(f"Error generating paraphrases with Qwen: {e}")
            # Fallback to back-translation
            return self.augment_with_paraphrasing(text, num_augmentations=min(num_paraphrases, 5))
    
    def augment_with_paraphrasing(self, text: str, num_augmentations: int = 2) -> List[str]:
        """
        Generate multiple paraphrases using back-translation
        More variations = more diverse paraphrases
        Uses multiple back-translation passes for diversity
        """
        augmented = []
        attempts = 0
        max_attempts = num_augmentations * 10  # Try many more times to get unique paraphrases
        
        # Use multiple back-translation passes for more diversity
        seen = set()
        seen.add(text.lower().strip())
        
        while len(augmented) < num_augmentations and attempts < max_attempts:
            # Multiple back-translation passes for more variation
            para = self.back_translate(text)
            if para:
                # Try another pass for more diversity
                para2 = self.back_translate(para)
                if para2:
                    para = para2
            
            para_normalized = para.lower().strip() if para else ""
            if para and para_normalized not in seen and len(para.strip()) >= 5:
                augmented.append(para)
                seen.add(para_normalized)
            attempts += 1
        
        # If we still don't have enough, use single back-translations
        while len(augmented) < num_augmentations and attempts < max_attempts * 2:
            para = self.back_translate(text)
            para_normalized = para.lower().strip() if para else ""
            if para and para_normalized not in seen and len(para.strip()) >= 5:
                augmented.append(para)
                seen.add(para_normalized)
            attempts += 1
        
        return augmented
    
    def augment(self, text: str, num_augmentations: int = 1, methods: List[str] = None) -> List[str]:
        """
        Generate augmented versions of text using sophisticated methods
        
        Args:
            text: Original Arabic text
            num_augmentations: Number of augmented versions to generate
            methods: List of methods to use ['back_translation', 'paraphrase']
        
        Returns:
            List of augmented texts
        """
        if methods is None:
            methods = ['back_translation']
        
        augmented = []
        
        for _ in range(num_augmentations):
            method = random.choice(methods)
            
            if method == 'back_translation':
                aug_text = self.back_translate(text)
            elif method == 'paraphrase':
                aug_text = self.paraphrase_via_backtranslation(text)
            else:
                aug_text = None
            
            if aug_text and len(aug_text.strip()) >= 5:  # Filter very short texts
                # Avoid duplicates
                if aug_text not in augmented and aug_text != text:
                    augmented.append(aug_text)
        
        return augmented if augmented else []


def balance_multilabel_dataset_selective(
    data: List[Dict],
    target_samples_per_class: int = None,
    augmenter: ArabicDataAugmenter = None,
    max_augmentations_per_sample: int = 20,  # Increased default to allow more augmentations
    use_paraphrasing: bool = True
) -> List[Dict]:
    """
    Selectively balance multi-label dataset by augmenting ONLY underrepresented classes
    to achieve equivalent representation. Well-represented classes are kept as-is.
    
    Args:
        data: List of samples with 'text' and 'labels' keys
        target_samples_per_class: Target count per class (default: median of well-represented classes)
        augmenter: ArabicDataAugmenter instance
        max_augmentations_per_sample: Maximum augmentations per original sample
        use_paraphrasing: Use paraphrasing for multiple variations
    
    Returns:
        Balanced dataset with selectively augmented samples
    """
    if augmenter is None:
        augmenter = ArabicDataAugmenter()
    
    # Count samples per label
    from collections import Counter
    label_counts = Counter()
    for sample in data:
        label_indices = [i for i, val in enumerate(sample['labels']) if val == 1]
        for idx in label_indices:
            label_counts[idx] += 1
    
    logger.info(f"Original label distribution: {dict(label_counts)}")
    
    # Calculate total samples
    total_samples = len(data)
    
    # Determine target count using formula: (1 - (class_count/total)) * lowest_well_represented
    # First, identify well-represented classes (above median)
    counts = sorted(label_counts.values())
    median = counts[len(counts) // 2]
    well_represented_labels = {idx: count for idx, count in label_counts.items() 
                              if count >= median}
    
    if not well_represented_labels:
        # Fallback: use median if no well-represented classes
        lowest_well_represented = median
        logger.warning("No well-represented classes found, using median as target")
    else:
        lowest_well_represented = min(well_represented_labels.values())
        logger.info(f"Lowest well-represented class count: {lowest_well_represented}")
    
    # Calculate target for each underrepresented class using the formula
    # target = ceil((1 - (class_count/total)) * lowest_well_represented)
    import math
    target_per_class = {}
    underrepresented_labels = {}
    
    for idx, count in label_counts.items():
        if count < median:  # Underrepresented
            class_distribution = count / total_samples
            target = math.ceil((1 - class_distribution) * lowest_well_represented)
            target_per_class[idx] = target
            underrepresented_labels[idx] = count
            logger.info(f"Label {idx}: {count} samples -> target = ceil((1 - {count}/{total_samples}) * {lowest_well_represented}) = ceil((1 - {class_distribution:.4f}) * {lowest_well_represented}) = {target}")
        else:
            # Well-represented: keep as-is
            target_per_class[idx] = count
    
    if target_samples_per_class is not None:
        # Override with provided target if specified
        logger.info(f"Using provided target: {target_samples_per_class} samples per class")
        for idx in underrepresented_labels:
            target_per_class[idx] = target_samples_per_class
    
    logger.info(f"Underrepresented labels: {list(underrepresented_labels.keys())}")
    logger.info(f"Well-represented labels (kept as-is): {list(well_represented_labels.keys())}")
    
    if not underrepresented_labels:
        logger.info("No underrepresented labels found. Returning original data.")
        return data.copy()
    
    # Start with original data
    augmented_data = data.copy()
    
    # Track current counts after augmentation
    current_counts = Counter(label_counts)
    
    # For each underrepresented label, augment samples to reach target
    # Strategy: Use samples with this label, but create augmented samples with ONLY the underrepresented label
    # This way we don't increase well-represented class counts
    for label_idx, current_count in underrepresented_labels.items():
        target = target_per_class[label_idx]
        needed = target - current_count
        if needed <= 0:
            continue
        
        logger.info(f"\nAugmenting label {label_idx}: {current_count} -> {target} (need +{needed})")
        
        # Find ALL samples with this underrepresented label
        samples_with_label = []
        for sample in data:
            label_indices = [i for i, val in enumerate(sample['labels']) if val == 1]
            if label_idx in label_indices:
                samples_with_label.append(sample)
        
        if not samples_with_label:
            logger.warning(f"No samples found for label {label_idx}")
            continue
        
        # Prioritize: samples with only underrepresented labels first
        samples_with_only_under = [s for s in samples_with_label if all(
            idx in underrepresented_labels for idx in [i for i, v in enumerate(s['labels']) if v == 1]
        )]
        samples_mixed = [s for s in samples_with_label if s not in samples_with_only_under]
        
        # Use only-underrepresented samples first, then mixed if needed
        samples_to_use = samples_with_only_under + samples_mixed
        
        logger.info(f"  Available: {len(samples_with_only_under)} only-underrepresented, {len(samples_mixed)} mixed")
        
        # Calculate augmentations needed per sample
        # Allow more augmentations if needed to reach target
        aug_per_sample = max(1, needed // len(samples_to_use) + 1)
        # Don't limit if we need more to reach target
        if aug_per_sample > max_augmentations_per_sample and needed > len(samples_to_use) * max_augmentations_per_sample:
            # We need more augmentations, so increase the limit
            aug_per_sample = max_augmentations_per_sample * 2  # Allow more
            logger.info(f"  Increasing augmentation limit to {aug_per_sample} per sample to reach target")
        else:
            aug_per_sample = min(aug_per_sample, max_augmentations_per_sample)
        
        logger.info(f"  Using {len(samples_to_use)} samples, ~{aug_per_sample} augmentations per sample")
        
        # Augment samples - loop through samples multiple times if needed to reach target
        generated = 0
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while generated < needed and iteration < max_iterations:
            iteration += 1
            logger.info(f"  Iteration {iteration}: Generated {generated}/{needed}, continuing...")
            
            for sample in samples_to_use:
                if generated >= needed:
                    break
                
                # Calculate how many more we need from this sample
                remaining_needed = needed - generated
                # Generate multiple augmentations per iteration
                num_aug = min(max_augmentations_per_sample, remaining_needed)
                
                if use_paraphrasing:
                    # Use Qwen for underrepresented labels (10-20 max paraphrases)
                    aug_texts = augmenter.paraphrase_with_qwen(
                        sample['text'],
                        num_paraphrases=min(num_aug, 20),  # Max 20 as requested
                        max_paraphrases=20
                    )
                else:
                    aug_texts = augmenter.augment(
                        sample['text'], 
                        num_augmentations=num_aug,
                        methods=['back_translation']
                    )
                
                # Add augmentations - CRITICAL: Only include the underrepresented label we're targeting
                # This prevents increasing well-represented class counts
                for aug_text in aug_texts:
                    if generated >= needed:
                        break
                    
                    # Create augmented sample with ONLY the underrepresented label we're targeting
                    # Strip out well-represented labels to avoid increasing their counts
                    new_labels = [0] * len(sample['labels'])
                    new_labels[label_idx] = 1  # Only the target underrepresented label
                    new_label_indices = [label_idx]
                    
                    # But keep other underrepresented labels if present (they also need help)
                    for idx in [i for i, val in enumerate(sample['labels']) if val == 1]:
                        if idx in underrepresented_labels and idx != label_idx:
                            new_labels[idx] = 1
                            new_label_indices.append(idx)
                    
                    augmented_data.append({
                        'id': len(augmented_data),
                        'text': aug_text,
                        'labels': new_labels,
                        'label_indices': new_label_indices
                    })
                    generated += 1
                    
                    # Update counts only for underrepresented labels
                    for idx in new_label_indices:
                        current_counts[idx] += 1
            
            if generated < needed and iteration >= max_iterations:
                logger.warning(f"  Could not reach target after {max_iterations} iterations. Generated {generated}/{needed}")
                break
    
    # Count final distribution
    final_label_counts = Counter()
    for sample in augmented_data:
        label_indices = [i for i, val in enumerate(sample['labels']) if val == 1]
        for idx in label_indices:
            final_label_counts[idx] += 1
    
    logger.info(f"\nFinal label distribution: {dict(final_label_counts)}")
    logger.info(f"Original samples: {len(data)}, Augmented samples: {len(augmented_data)}")
    
    # Show improvement per label
    logger.info("\nLabel improvements:")
    for idx in sorted(final_label_counts.keys()):
        orig = label_counts.get(idx, 0)
        final = final_label_counts.get(idx, 0)
        if idx in underrepresented_labels:
            target = target_per_class.get(idx, orig)
            logger.info(f"  Label {idx}: {orig} -> {final} (+{final - orig}) [AUGMENTED to target {target}]")
        else:
            logger.info(f"  Label {idx}: {orig} -> {final} (kept as-is, no augmentation)")
    
    return augmented_data
