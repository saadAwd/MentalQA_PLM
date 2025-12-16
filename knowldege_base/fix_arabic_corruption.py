# fix_arabic_corruption.py
# Post-processing script to fix corrupted Arabic text from PDF extraction

import json
import re
import os
from config import PROCESSED_DIR


def fix_arabic_corruption(text):
    """
    Fix common Arabic corruption patterns from PDF extraction.
    
    Common issues:
    - 'اU' should be 'الأ' (alif-lam before words starting with alif)
    - 'Uن' should be 'أن' or 'إن' (conditional particles)
    - '%' in 'ا%وﻟﻴﺔ' should be 'الأولية'
    - 'Uﺷ' should be 'أش' or 'إش'
    - 'Uز' should be 'أز' or 'إز'
    - 'Uﻛ' should be 'أك' or 'إك'
    """
    if not text:
        return text
    
    # Make a copy to avoid modifying original
    fixed = text
    
    # Pattern 1: 'اU' → 'الأ' (most common - 592 instances)
    # This is 'al-' prefix corrupted
    # The U character (U+0055) is being used instead of proper Arabic alif
    # We need to match 'ا' (Arabic alif) + 'U' (Latin U) + Arabic character
    # Handle all cases including those with diacritics/formatting
    
    # First, handle specific common words
    fixed = re.sub(r'اUﺣـ?ﻮال', 'الأحوال', fixed)
    fixed = re.sub(r'اUﺷـ?ﺨﺎص', 'الأشخاص', fixed)
    fixed = re.sub(r'اUوﻟﻴـ?ﺔ', 'الأولية', fixed)
    fixed = re.sub(r'اUزﻣـ?ﺎت', 'الأزمات', fixed)
    fixed = re.sub(r'اUﻛﺜـ?ﺮ', 'أكثر', fixed)
    fixed = re.sub(r'اUﻟﻔـ?ﺔ', 'الألفة', fixed)
    
    # General pattern: 'ا' + 'U' + any Arabic character (with optional formatting)
    # Match Arabic alif + Latin U + Arabic character (may have diacritics/formatting)
    # Handle formatting characters (ـ) that might appear between U and Arabic char
    fixed = re.sub(r'اU[ـ\s]*([\u0600-\u06FF])', r'الأ\1', fixed)
    
    # Multiple passes to catch all cases
    for _ in range(2):
        fixed = re.sub(r'اU([\u0600-\u06FF])', r'الأ\1', fixed)
    
    # Also handle cases where there might be spaces between
    fixed = re.sub(r'اU\s+([\u0600-\u06FF])', r'الأ \1', fixed)
    
    # Pattern 2: 'Uن' → 'أن' (conditional 'if')
    # Context: usually at start of sentences or after 'إذا'
    fixed = re.sub(r'Uن\s', r'أن ', fixed)
    fixed = re.sub(r'Uن([\u0600-\u06FF])', r'أن\1', fixed)
    
    # Pattern 3: '%' in 'ا%وﻟﻴﺔ' → 'الأولية'
    # Specific fix for this common word
    fixed = re.sub(r'ا%وﻟﻴﺔ', 'الأولية', fixed)
    fixed = re.sub(r'ا%([\u0600-\u06FF])', r'الأ\1', fixed)  # General case
    
    # Pattern 4: 'Uﺷ' → 'أش' (usually 'أشخاص' - people)
    # Check if followed by 'خاص' to make it 'أشخاص'
    fixed = re.sub(r'Uﺷخاص', 'أشخاص', fixed)
    fixed = re.sub(r'Uﺷ([\u0600-\u06FF])', r'أش\1', fixed)
    
    # Pattern 5: 'Uز' → 'أز' 
    fixed = re.sub(r'Uز([\u0600-\u06FF])', r'أز\1', fixed)
    
    # Pattern 6: 'Uﻛ' → 'أك' or 'إك'
    # Usually 'أكثر' (more/most)
    fixed = re.sub(r'Uﻛثر', 'أكثر', fixed)
    fixed = re.sub(r'Uﻛ([\u0600-\u06FF])', r'أك\1', fixed)
    
    # Pattern 7: 'Uم' → 'أم' or 'إم'
    fixed = re.sub(r'Uم([\u0600-\u06FF])', r'أم\1', fixed)
    
    # Pattern 8: 'Uل' → 'أل' or 'إل'
    fixed = re.sub(r'Uل([\u0600-\u06FF])', r'أل\1', fixed)
    
    # Pattern 9: 'Uت' → 'أت' or 'إت'
    fixed = re.sub(r'Uت([\u0600-\u06FF])', r'أت\1', fixed)
    
    # Pattern 10: 'Uد' → 'أد' or 'إد'
    fixed = re.sub(r'Uد([\u0600-\u06FF])', r'أد\1', fixed)
    
    # Pattern 11: 'Uر' → 'أر' or 'إر'
    fixed = re.sub(r'Uر([\u0600-\u06FF])', r'أر\1', fixed)
    
    # Pattern 12: 'Uب' → 'أب' or 'إب'
    fixed = re.sub(r'Uب([\u0600-\u06FF])', r'أب\1', fixed)
    
    # Pattern 13: 'Uف' → 'أف' or 'إف'
    fixed = re.sub(r'Uف([\u0600-\u06FF])', r'أف\1', fixed)
    
    # Pattern 14: 'Uق' → 'أق' or 'إق'
    fixed = re.sub(r'Uق([\u0600-\u06FF])', r'أق\1', fixed)
    
    # Pattern 15: 'Uع' → 'أع' or 'إع'
    fixed = re.sub(r'Uع([\u0600-\u06FF])', r'أع\1', fixed)
    
    # Pattern 16: 'Uص' → 'أص' or 'إص'
    fixed = re.sub(r'Uص([\u0600-\u06FF])', r'أص\1', fixed)
    
    # Pattern 17: 'Uه' → 'أه' or 'إه'
    fixed = re.sub(r'Uه([\u0600-\u06FF])', r'أه\1', fixed)
    
    # Pattern 18: 'Uي' → 'أي' or 'إي'
    fixed = re.sub(r'Uي([\u0600-\u06FF])', r'أي\1', fixed)
    
    # Pattern 19: 'Uو' → 'أو' or 'إو'
    fixed = re.sub(r'Uو([\u0600-\u06FF])', r'أو\1', fixed)
    
    # Pattern 20: 'Uا' → 'أا' or 'إا' (less common, might be 'أ' + 'ا')
    fixed = re.sub(r'Uا([\u0600-\u06FF])', r'أ\1', fixed)
    
    # Clean up any remaining standalone 'U' before Arabic characters
    fixed = re.sub(r'U([\u0600-\u06FF])', r'أ\1', fixed)
    
    return fixed


def fix_book_file(input_file, output_file=None):
    """Fix corruption in a book JSONL file."""
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_fixed.jsonl')
    
    print(f"\n{'='*70}")
    print(f"FIXING ARABIC CORRUPTION")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    fixed_count = 0
    total_chars = 0
    fixed_chars = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            original_text = doc.get('clean_text', '')
            
            if original_text:
                total_chars += len(original_text)
                fixed_text = fix_arabic_corruption(original_text)
                fixed_chars += len(fixed_text)
                
                # Count fixes
                if fixed_text != original_text:
                    fixed_count += 1
                    # Count how many patterns were fixed
                    patterns_fixed = sum([
                        original_text.count('اU') - fixed_text.count('اU'),
                        original_text.count('Uن') - fixed_text.count('Uن'),
                        original_text.count('%') - fixed_text.count('%'),
                    ])
                
                doc['clean_text'] = fixed_text
                doc['text_length'] = len(fixed_text)
                doc['corruption_fixed'] = True
            
            outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*70}")
    print("FIXING COMPLETE")
    print(f"{'='*70}")
    print(f"Documents processed: {line_num}")
    print(f"Documents with fixes: {fixed_count}")
    print(f"Total characters: {total_chars:,}")
    print(f"Fixed file saved to: {output_file}")
    
    return output_file


def fix_chunks_file(input_file, output_file=None):
    """Fix corruption in a chunks JSONL file."""
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_fixed.jsonl')
    
    print(f"\n{'='*70}")
    print(f"FIXING ARABIC CORRUPTION IN CHUNKS")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    fixed_count = 0
    total_chunks = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            chunk = json.loads(line)
            original_text = chunk.get('text', '')
            
            if original_text:
                total_chunks += 1
                fixed_text = fix_arabic_corruption(original_text)
                
                if fixed_text != original_text:
                    fixed_count += 1
                
                chunk['text'] = fixed_text
                chunk['text_length'] = len(fixed_text)
                chunk['corruption_fixed'] = True
            
            outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*70}")
    print("FIXING COMPLETE")
    print(f"{'='*70}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with fixes: {fixed_count}")
    print(f"Fixed file saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import sys
    
    # Prefer full books file if it exists, otherwise fall back to POC file
    books_candidates = [
        os.path.join(PROCESSED_DIR, "books_all.jsonl"),
        os.path.join(PROCESSED_DIR, "books_poc.jsonl"),
    ]
    books_file = None
    for candidate in books_candidates:
        if os.path.exists(candidate):
            books_file = candidate
            break
    
    if books_file:
        fix_book_file(books_file)
    
    # Fix chunks if they exist
    chunks_file = os.path.join(PROCESSED_DIR, "book_chunks_example.jsonl")
    if os.path.exists(chunks_file):
        fix_chunks_file(chunks_file)
    
    print("\n✅ Arabic corruption fixing complete!")
    print("\nNext steps:")
    print("1. Review the fixed files")
    print("2. Compare before/after to verify improvements")
    print("3. Use the fixed files for embedding generation")

