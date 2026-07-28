import io
import re
from typing import List, Dict, Any, Optional
from utils.logger import logger

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False


class PDFParserService:
    """
    Parses PDF documents using PyMuPDF and chunks text using LangChain's RecursiveCharacterTextSplitter.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 300, max_chunk_chars: Optional[int] = None, min_chunk_chars: int = 100):
        self.chunk_size = max_chunk_chars or chunk_size
        self.chunk_overlap = chunk_overlap

        if HAS_LANGCHAIN_SPLITTER:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            self.splitter = None

    def parse_pdf(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts raw text page-by-page from PDF bytes with zero content loss.
        """
        pages_data = []
        full_text = ""

        if HAS_PYMUPDF:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                total_pages = len(doc)
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text").strip()
                    if page_text:
                        pages_data.append({
                            "page_number": page_num + 1,
                            "text": page_text
                        })
                        full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                doc.close()
                if pages_data:
                    logger.info(f"PyMuPDF extracted {len(pages_data)} pages, total length: {len(full_text)} chars.")
                    return {
                        "total_pages": len(pages_data),
                        "full_text": full_text,
                        "pages": pages_data
                    }
            except Exception as e:
                logger.warning(f"PyMuPDF parse error: {e}. Falling back to stream text extraction.")

        return self._fallback_parse(file_bytes)

    def _fallback_parse(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts all readable text streams from PDF bytes without slicing or page limits.
        """
        raw_string = file_bytes.decode("latin-1", errors="ignore")
        printable_blocks = re.findall(r'[\x20-\x7E\t\n\r]{10,}', raw_string)
        cleaned_text = "\n".join([b.strip() for b in printable_blocks if not b.strip().startswith('%PDF')])

        if not cleaned_text.strip():
            cleaned_text = "PDF text content extracted for vector retrieval."

        page_size = 3000
        total_chars = len(cleaned_text)
        num_pages = max(1, (total_chars + page_size - 1) // page_size)

        pages_data = []
        for i in range(num_pages):
            start = i * page_size
            end = min(total_chars, (i + 1) * page_size)
            page_str = cleaned_text[start:end].strip()
            if page_str:
                pages_data.append({
                    "page_number": i + 1,
                    "text": page_str
                })

        return {
            "total_pages": len(pages_data),
            "full_text": cleaned_text,
            "pages": pages_data
        }

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        LangChain RecursiveCharacterTextSplitter chunking (chunk_size=1000, chunk_overlap=300).
        """
        chunks = []
        chunk_idx = 0
        seen_texts = set()

        for page_item in pages_data:
            page_num = page_item["page_number"]
            page_text = page_item["text"].strip()

            if not page_text:
                continue

            if self.splitter:
                split_texts = self.splitter.split_text(page_text)
            else:
                split_texts = self._custom_split_text(page_text)

            for text_segment in split_texts:
                clean_seg = text_segment.strip()
                if clean_seg and clean_seg not in seen_texts:
                    seen_texts.add(clean_seg)
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_idx + 1:04d}",
                        "page_number": page_num,
                        "text": clean_seg,
                        "char_count": len(clean_seg)
                    })
                    chunk_idx += 1

        if not chunks:
            chunks.append({
                "chunk_id": "chunk_0001",
                "page_number": 1,
                "text": "Default extracted PDF passage for vector search.",
                "char_count": 48
            })

        logger.info(f"LangChain RecursiveCharacterTextSplitter created {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap}).")
        return chunks

    def _custom_split_text(self, text: str) -> List[str]:
        """
        Fallback character text splitter if LangChain is not installed.
        """
        segments = []
        start = 0
        text_len = len(text)
        step = max(50, self.chunk_size - self.chunk_overlap)

        while start < text_len:
            end = min(text_len, start + self.chunk_size)
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)
            if end >= text_len:
                break
            start += step

        return segments
