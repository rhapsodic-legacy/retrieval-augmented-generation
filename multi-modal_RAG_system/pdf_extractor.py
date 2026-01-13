"""
PDF Extractor with Layout Awareness

Extracts:
- Text with layout preservation
- Images with bounding boxes
- Tables as structured data
- Charts/diagrams
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import io
import re

from .content_types import (
    TextChunk,
    ImageContent,
    TableContent,
    PDFPage,
    BoundingBox,
    generate_content_id,
    ContentType,
)


@dataclass
class PDFExtractionConfig:
    """Configuration for PDF extraction."""
    
    # Text extraction
    extract_text: bool = True
    preserve_layout: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Image extraction
    extract_images: bool = True
    min_image_size: int = 50  # Minimum width/height in pixels
    
    # Table extraction
    extract_tables: bool = True
    
    # Page rendering
    render_pages: bool = False
    render_dpi: int = 150


class PDFExtractor:
    """
    Extracts content from PDFs with layout awareness.
    
    Uses multiple libraries for best results:
    - pdfplumber for text and tables
    - PyMuPDF (fitz) for images and rendering
    - pdf2image for page rendering
    """
    
    def __init__(self, config: Optional[PDFExtractionConfig] = None):
        self.config = config or PDFExtractionConfig()
    
    def extract(self, file_path: str) -> list[PDFPage]:
        """Extract all content from a PDF."""
        file_path = Path(file_path)
        
        pages = []
        
        # Try pdfplumber first (best for text and tables)
        try:
            pages = self._extract_with_pdfplumber(file_path)
        except ImportError:
            # Fall back to PyMuPDF
            try:
                pages = self._extract_with_pymupdf(file_path)
            except ImportError:
                raise ImportError(
                    "Install pdfplumber or PyMuPDF (fitz) for PDF extraction: "
                    "pip install pdfplumber pymupdf"
                )
        
        # Extract images with PyMuPDF (better image extraction)
        if self.config.extract_images:
            try:
                self._extract_images_pymupdf(file_path, pages)
            except ImportError:
                pass
        
        return pages
    
    def _extract_with_pdfplumber(self, file_path: Path) -> list[PDFPage]:
        """Extract using pdfplumber."""
        import pdfplumber
        
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                pdf_page = PDFPage(
                    id=generate_content_id(str(file_path), ContentType.PDF_PAGE, page_num),
                    content_type=ContentType.PDF_PAGE,
                    source_file=str(file_path),
                    text="",
                    page_number=page_num + 1,
                    width=page.width,
                    height=page.height,
                )
                
                # Extract text blocks
                if self.config.extract_text:
                    text_blocks = self._extract_text_blocks(page, page_num, file_path)
                    pdf_page.text_blocks = text_blocks
                    pdf_page.text = "\n\n".join(block.text for block in text_blocks)
                
                # Extract tables
                if self.config.extract_tables:
                    tables = self._extract_tables(page, page_num, file_path)
                    pdf_page.tables = tables
                
                pages.append(pdf_page)
        
        return pages
    
    def _extract_text_blocks(
        self,
        page,
        page_num: int,
        file_path: Path,
    ) -> list[TextChunk]:
        """Extract text blocks from a pdfplumber page."""
        blocks = []
        
        # Get words with positions
        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3,
        )
        
        if not words:
            return blocks
        
        # Group words into lines
        lines = self._group_words_into_lines(words)
        
        # Group lines into paragraphs
        paragraphs = self._group_lines_into_paragraphs(lines)
        
        # Chunk paragraphs
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_text = " ".join(para["text"])
            para_length = len(para_text)
            
            if current_length + para_length > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                blocks.append(TextChunk(
                    id=generate_content_id(str(file_path), ContentType.TEXT, page_num, chunk_index),
                    content_type=ContentType.TEXT,
                    source_file=str(file_path),
                    text=chunk_text,
                    page_number=page_num + 1,
                    chunk_index=chunk_index,
                    bbox=para.get("bbox"),
                ))
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-self.config.chunk_overlap:] if len(chunk_text) > self.config.chunk_overlap else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
                chunk_index += 1
            
            current_chunk.append(para_text)
            current_length += para_length
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            blocks.append(TextChunk(
                id=generate_content_id(str(file_path), ContentType.TEXT, page_num, chunk_index),
                content_type=ContentType.TEXT,
                source_file=str(file_path),
                text=chunk_text,
                page_number=page_num + 1,
                chunk_index=chunk_index,
            ))
        
        return blocks
    
    def _group_words_into_lines(self, words: list[dict]) -> list[dict]:
        """Group words into lines based on y-position."""
        if not words:
            return []
        
        # Sort by y, then x
        sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]["top"]
        
        for word in sorted_words[1:]:
            # Same line if y is close
            if abs(word["top"] - current_y) < 5:
                current_line.append(word)
            else:
                lines.append({
                    "words": current_line,
                    "text": " ".join(w["text"] for w in current_line),
                    "top": current_y,
                    "x0": min(w["x0"] for w in current_line),
                    "x1": max(w["x1"] for w in current_line),
                })
                current_line = [word]
                current_y = word["top"]
        
        # Don't forget last line
        if current_line:
            lines.append({
                "words": current_line,
                "text": " ".join(w["text"] for w in current_line),
                "top": current_y,
                "x0": min(w["x0"] for w in current_line),
                "x1": max(w["x1"] for w in current_line),
            })
        
        return lines
    
    def _group_lines_into_paragraphs(self, lines: list[dict]) -> list[dict]:
        """Group lines into paragraphs based on spacing."""
        if not lines:
            return []
        
        paragraphs = []
        current_para = [lines[0]]
        
        for i, line in enumerate(lines[1:], 1):
            prev_line = lines[i - 1]
            
            # Calculate vertical gap
            gap = line["top"] - prev_line["top"]
            
            # Detect heading (larger font = more spacing)
            is_heading = len(line["text"]) < 100 and gap > 20
            
            # New paragraph if large gap or heading
            if gap > 15 or is_heading:
                paragraphs.append({
                    "text": [l["text"] for l in current_para],
                    "bbox": BoundingBox(
                        x0=min(l["x0"] for l in current_para),
                        y0=current_para[0]["top"],
                        x1=max(l["x1"] for l in current_para),
                        y1=current_para[-1]["top"] + 10,
                    ),
                })
                current_para = [line]
            else:
                current_para.append(line)
        
        # Don't forget last paragraph
        if current_para:
            paragraphs.append({
                "text": [l["text"] for l in current_para],
                "bbox": BoundingBox(
                    x0=min(l["x0"] for l in current_para),
                    y0=current_para[0]["top"],
                    x1=max(l["x1"] for l in current_para),
                    y1=current_para[-1]["top"] + 10,
                ) if current_para else None,
            })
        
        return paragraphs
    
    def _extract_tables(
        self,
        page,
        page_num: int,
        file_path: Path,
    ) -> list[TableContent]:
        """Extract tables from a pdfplumber page."""
        tables = []
        
        # Find tables
        page_tables = page.find_tables()
        
        for i, table in enumerate(page_tables):
            # Extract table data
            data = table.extract()
            
            if not data or len(data) < 2:
                continue
            
            # First row as headers
            headers = [str(h) if h else f"col_{j}" for j, h in enumerate(data[0])]
            rows = data[1:]
            
            # Clean rows
            clean_rows = []
            for row in rows:
                clean_row = [str(cell) if cell else "" for cell in row]
                clean_rows.append(clean_row)
            
            # Create table content
            table_content = TableContent(
                id=generate_content_id(str(file_path), ContentType.TABLE, page_num, i),
                content_type=ContentType.TABLE,
                source_file=str(file_path),
                text="",  # Will be generated
                headers=headers,
                rows=clean_rows,
                bbox=BoundingBox(
                    x0=table.bbox[0],
                    y0=table.bbox[1],
                    x1=table.bbox[2],
                    y1=table.bbox[3],
                    page=page_num,
                ),
                metadata={"page": page_num + 1},
            )
            
            tables.append(table_content)
        
        return tables
    
    def _extract_with_pymupdf(self, file_path: Path) -> list[PDFPage]:
        """Extract using PyMuPDF (fitz)."""
        import fitz
        
        pages = []
        
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]
            
            text_content = []
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = " ".join(span["text"] for span in line.get("spans", []))
                        text_content.append(line_text)
            
            pdf_page = PDFPage(
                id=generate_content_id(str(file_path), ContentType.PDF_PAGE, page_num),
                content_type=ContentType.PDF_PAGE,
                source_file=str(file_path),
                text="\n".join(text_content),
                page_number=page_num + 1,
                width=page.rect.width,
                height=page.rect.height,
            )
            
            # Create text chunk for the whole page
            if text_content:
                pdf_page.text_blocks = [TextChunk(
                    id=generate_content_id(str(file_path), ContentType.TEXT, page_num, 0),
                    content_type=ContentType.TEXT,
                    source_file=str(file_path),
                    text="\n".join(text_content),
                    page_number=page_num + 1,
                    chunk_index=0,
                )]
            
            pages.append(pdf_page)
        
        doc.close()
        return pages
    
    def _extract_images_pymupdf(self, file_path: Path, pages: list[PDFPage]):
        """Extract images using PyMuPDF."""
        import fitz
        
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            if page_num >= len(pages):
                break
            
            pdf_page = pages[page_num]
            images = []
            
            # Get images on page
            image_list = page.get_images()
            
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Check size
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    if width < self.config.min_image_size or height < self.config.min_image_size:
                        continue
                    
                    # Determine format
                    ext = base_image.get("ext", "png")
                    
                    image_content = ImageContent(
                        id=generate_content_id(str(file_path), ContentType.IMAGE, page_num, img_idx),
                        content_type=ContentType.IMAGE,
                        source_file=str(file_path),
                        text=f"Image on page {page_num + 1}",
                        image_data=image_bytes,
                        width=width,
                        height=height,
                        format=ext,
                        metadata={"page": page_num + 1},
                    )
                    
                    images.append(image_content)
                    
                except Exception as e:
                    print(f"Warning: Failed to extract image {img_idx} from page {page_num}: {e}")
            
            pdf_page.images = images
        
        doc.close()
    
    def render_page(self, file_path: str, page_num: int) -> bytes:
        """Render a page as an image."""
        try:
            import fitz
            
            doc = fitz.open(file_path)
            page = doc[page_num]
            
            # Render at configured DPI
            zoom = self.config.render_dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            image_bytes = pix.tobytes("png")
            doc.close()
            
            return image_bytes
            
        except ImportError:
            raise ImportError("Install PyMuPDF for page rendering: pip install pymupdf")


def extract_pdf(
    file_path: str,
    config: Optional[PDFExtractionConfig] = None,
) -> list[PDFPage]:
    """Extract content from a PDF file."""
    extractor = PDFExtractor(config)
    return extractor.extract(file_path)
