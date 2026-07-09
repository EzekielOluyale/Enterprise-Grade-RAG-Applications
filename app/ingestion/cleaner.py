import re

def clean_amusEcode_data(raw_text: str) -> str:
    """
    A production-grade text cleaner for corporate PDF extractions.
    Strips repetitive layout noise, boilerplate, and normalizes formatting
    so the vector database only receives high-signal 'True Data'.
    """

    cleaned_text = raw_text

    # Removes standard "Page 1", "Page 1 of 10", "pg 4" 
    # (?i) makes it case-insensitive
    cleaned_text = re.sub(r"(?i)\b(?:page|pg)\s*\d+(?:\s*of\s*\d+)?\b", "", cleaned_text)
    
    # Removes standalone page numbers formatted like "- 1 -" or " 12 " on their own line
    cleaned_text = re.sub(r"^\s*[-]?\s*\d+\s*[-]?\s*$", "", cleaned_text, flags=re.MULTILINE)

    # Removes entire lines that contain the word "Confidential"
    cleaned_text = re.sub(r"(?i)^.*confidential.*$", "", cleaned_text, flags=re.MULTILINE)
    
    # Removes Copyright notices (e.g., "© 2026 amusEcode", "Copyright 2026")
    cleaned_text = re.sub(r"(?i)(?:©|copyright)\s*\d{4}.*?(?=\n|$)", "", cleaned_text)
    
    # Removes specific repeating document headers
    cleaned_text = re.sub(r"(?i)amusEcode policy manual", "", cleaned_text)

    # Removes Table of Contents dot leaders (e.g., "Introduction ............... 4")
    cleaned_text = re.sub(r"\.{4,}\s*\d+", "", cleaned_text)
    
    # Removes horizontal divider lines made of dashes or underscores (e.g., "---------")
    cleaned_text = re.sub(r"[-_]{4,}", "", cleaned_text)
    
    # Removes URLs and email addresses (Optional: remove if you want strict policy text only)
    cleaned_text = re.sub(r"http[s]?://\S+|www\.\S+|\S+@\S+\.\S+", "", cleaned_text)

    # Replace weird unicode spaces or multiple horizontal spaces with a single normal space
    cleaned_text = re.sub(r"[^\S\n]+", " ", cleaned_text)
    
    # Strip leading/trailing spaces from every individual line
    cleaned_text = re.sub(r"^[ \t]+|[ \t]+$", "", cleaned_text, flags=re.MULTILINE)
    
    # Collapse 3 or more blank lines into exactly 2 blank lines (preserves paragraphs)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # Return the final string, stripping any loose space at the very beginning or end
    return cleaned_text.strip()