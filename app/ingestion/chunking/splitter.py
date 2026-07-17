from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Uses RecursiveCharacterTextSplitter to ensure chunks are always 
    under the limit, even if paragraphs are massive.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip(): 
            return []
            
        # Initialize the splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks