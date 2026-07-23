from typing import List
import logfire
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_text(text: str, source_name: str = "document.pdf", chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Uses RecursiveCharacterTextSplitter to ensure chunks are always 
    under the limit, even if paragraphs are massive.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text), source=source_name):
        if not text.strip():
            logfire.warn("⚠️ Received empty text for chunking", source=source_name) 
            return []

        # Split semantically by Markdown headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(text)

        logfire.info("📑 Markdown header splitting completed", header_splits_count=len(header_splits))
            
        # Use Recursive Character Splitter as a safety net
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(header_splits)

        for chunk in chunks:
            chunk.metadata["source"] = source_name

        # Drop any empty or whitespace-only chunks
        chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
        
        logfire.info(
            "✅ Text chunking finished successfully", 
            final_chunks_count=len(chunks), 
            source=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return chunks