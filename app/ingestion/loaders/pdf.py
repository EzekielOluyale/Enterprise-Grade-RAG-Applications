from pathlib import Path

import logfire
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def parse_pdf(file_path: str | Path) -> str:
    """
    Parse a PDF using Docling.

    Features:
        • Text extraction
        • Table extraction
        • Reading order preservation
        • Image understanding hooks
        • Layout understanding
        • Scanned PDF support (OCR when needed)

    Returns
    -------
    str
        Markdown representation of the document.
    """

    file_path = Path(file_path)

    with logfire.span(
        "PDF Parsing",
        filename=file_path.name,
    ):
        try:
            logfire.info(f"Parsing {file_path.name}")

            result = converter.convert(file_path)

            # Convert the document into Markdown
            markdown = result.document.export_to_markdown()

            if not markdown.strip():
                logfire.warning("No content extracted from PDF.")
            else:
                logfire.info(
                    f"Successfully extracted {len(markdown):,} characters."
                )

            return markdown

        except Exception as e:
            logfire.exception(
                f"Failed to parse PDF '{file_path.name}': {e}"
            )
            raise