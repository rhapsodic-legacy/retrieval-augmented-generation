"""
Multi-Modal RAG System

Unified RAG across text, images, PDFs, and tables.

Features:
- Process PDFs with layout awareness
- Extract and describe images
- Handle tables as structured data
- Query across all modalities
"""

from dataclasses import dataclass, field
from typing import Optional, Generator, Union
from pathlib import Path
import mimetypes

from .providers import create_llm, create_embedding, BaseLLM, BaseEmbedding, ImageInput
from .providers.embeddings import VisionDescriptionEmbedding
from .extraction import (
    ContentType,
    ContentItem,
    TextChunk,
    ImageContent,
    TableContent,
    ChartContent,
    PDFPage,
    PDFExtractor,
    PDFExtractionConfig,
    ImageProcessor,
    ImageProcessingConfig,
    TableExtractor,
    TableExtractionConfig,
)
from .indexing import MultimodalIndex, SearchResult


@dataclass
class MultimodalRAGConfig:
    """Configuration for Multi-Modal RAG."""
    
    # LLM settings
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    
    # Embedding settings
    embedding_provider: str = "local"
    embedding_model: Optional[str] = None
    
    # PDF extraction
    extract_pdf_images: bool = True
    extract_pdf_tables: bool = True
    pdf_chunk_size: int = 1000
    
    # Image processing
    generate_image_descriptions: bool = True
    extract_chart_data: bool = True
    
    # Search
    n_results: int = 5
    
    # Generation
    max_tokens: int = 2000
    temperature: float = 0.1


@dataclass
class MultimodalContext:
    """Context retrieved for a query."""
    text_results: list[SearchResult] = field(default_factory=list)
    image_results: list[SearchResult] = field(default_factory=list)
    table_results: list[SearchResult] = field(default_factory=list)
    
    @property
    def all_results(self) -> list[SearchResult]:
        return self.text_results + self.image_results + self.table_results
    
    @property
    def sources(self) -> list[str]:
        return list(set(r.item.source_file for r in self.all_results))


@dataclass
class MultimodalResponse:
    """Response from Multi-Modal RAG."""
    answer: str
    context: MultimodalContext
    query: str
    model: str
    
    # Images included in response
    images: list[ImageContent] = field(default_factory=list)
    
    # Tables included
    tables: list[TableContent] = field(default_factory=list)
    
    metadata: dict = field(default_factory=dict)


class MultimodalRAG:
    """
    Multi-Modal RAG System.
    
    Handles:
    - Text documents
    - PDFs with layout awareness
    - Images with vision processing
    - Tables as structured data
    - Charts with data extraction
    
    Usage:
        rag = MultimodalRAG()
        
        # Add documents
        rag.add_pdf("report.pdf")
        rag.add_image("chart.png")
        rag.add_table("data.csv")
        
        # Query across all modalities
        response = rag.query("Find the chart showing Q3 revenue")
        print(response.answer)
        
        # Access images in response
        for img in response.images:
            display(img.image_data)
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on multimodal content including text, images, tables, and charts.

When answering:
1. Reference specific documents, images, or tables
2. For charts/graphs, describe the data and trends
3. For tables, summarize key information
4. Use exact numbers and values when available
5. Indicate which source each piece of information comes from

Format tables using markdown when presenting tabular data."""

    QUERY_PROMPT = """Based on the following multimodal context, answer the question.

{context}

QUESTION: {question}

Provide a comprehensive answer referencing the relevant content."""

    IMAGE_CONTEXT_TEMPLATE = """[IMAGE: {description}]
Source: {source}
Type: {image_type}
{chart_info}"""

    TABLE_CONTEXT_TEMPLATE = """[TABLE: {caption}]
Source: {source}
Columns: {columns}
Rows: {num_rows}
Sample:
{sample}"""

    def __init__(self, config: Optional[MultimodalRAGConfig] = None):
        """Initialize Multi-Modal RAG."""
        self.config = config or MultimodalRAGConfig()
        
        # Initialize LLM
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        # Initialize embeddings
        self.text_embeddings = create_embedding(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Create vision-based embeddings for images
        self.multimodal_embeddings = VisionDescriptionEmbedding(
            text_embedding=self.text_embeddings,
            llm=self.llm,
        )
        
        # Initialize index
        self.index = MultimodalIndex(
            embedding_provider=self.text_embeddings,
        )
        
        # Initialize extractors
        self.pdf_extractor = PDFExtractor(PDFExtractionConfig(
            extract_images=self.config.extract_pdf_images,
            extract_tables=self.config.extract_pdf_tables,
            chunk_size=self.config.pdf_chunk_size,
        ))
        
        self.image_processor = ImageProcessor(
            llm=self.llm,
            config=ImageProcessingConfig(
                generate_descriptions=self.config.generate_image_descriptions,
                extract_chart_data=self.config.extract_chart_data,
            ),
        )
        
        self.table_extractor = TableExtractor()
        
        # Track sources
        self.sources: dict[str, dict] = {}
    
    # =========================================================================
    # Adding Content
    # =========================================================================
    
    def add_file(self, file_path: str) -> dict:
        """Add a file based on its type."""
        path = Path(file_path)
        ext = path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if ext == ".pdf":
            return self.add_pdf(file_path)
        elif ext in (".csv", ".tsv", ".xlsx", ".xls"):
            return self.add_table(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            return self.add_image(file_path)
        elif ext in (".txt", ".md"):
            return self.add_text_file(file_path)
        elif ext == ".html":
            return self.add_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def add_pdf(self, file_path: str) -> dict:
        """Add a PDF document."""
        pages = self.pdf_extractor.extract(file_path)
        
        stats = {"pages": 0, "text_chunks": 0, "images": 0, "tables": 0}
        
        for page in pages:
            stats["pages"] += 1
            
            # Add text chunks
            for chunk in page.text_blocks:
                chunk.embedding = self.text_embeddings.embed_text(chunk.text)
                self.index.add(chunk)
                stats["text_chunks"] += 1
            
            # Process and add images
            for image in page.images:
                try:
                    processed = self.image_processor.process_image(
                        image.image_data,
                        file_path,
                        f"image/{image.format}",
                    )
                    processed.embedding = self.text_embeddings.embed_text(processed.text)
                    self.index.add(processed)
                    stats["images"] += 1
                except Exception as e:
                    print(f"Warning: Failed to process image: {e}")
            
            # Add tables
            for table in page.tables:
                table.embedding = self.text_embeddings.embed_text(table.text)
                self.index.add(table)
                stats["tables"] += 1
        
        self.sources[file_path] = {"type": "pdf", "stats": stats}
        return stats
    
    def add_image(self, file_path: str) -> dict:
        """Add an image file."""
        processed = self.image_processor.process_image_file(file_path)
        processed.embedding = self.text_embeddings.embed_text(processed.text)
        self.index.add(processed)
        
        stats = {"images": 1, "type": processed.image_type}
        self.sources[file_path] = {"type": "image", "stats": stats}
        return stats
    
    def add_table(self, file_path: str) -> dict:
        """Add a table file (CSV, Excel, etc.)."""
        tables = self.table_extractor.extract_file(file_path)
        
        for table in tables:
            table.embedding = self.text_embeddings.embed_text(table.text)
            self.index.add(table)
        
        stats = {"tables": len(tables)}
        self.sources[file_path] = {"type": "table", "stats": stats}
        return stats
    
    def add_text_file(self, file_path: str) -> dict:
        """Add a text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.add_text(content, file_path)
    
    def add_text(
        self,
        text: str,
        source: str = "text",
        chunk_size: int = 1000,
    ) -> dict:
        """Add raw text."""
        # Simple chunking
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Add chunks
        for i, chunk_text in enumerate(chunks):
            chunk = TextChunk(
                id=f"{source}::chunk_{i}",
                content_type=ContentType.TEXT,
                source_file=source,
                text=chunk_text,
                chunk_index=i,
            )
            chunk.embedding = self.text_embeddings.embed_text(chunk_text)
            self.index.add(chunk)
        
        stats = {"text_chunks": len(chunks)}
        self.sources[source] = {"type": "text", "stats": stats}
        return stats
    
    def add_html(self, file_path: str) -> dict:
        """Add an HTML file (extracts tables and text)."""
        from bs4 import BeautifulSoup
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        soup = BeautifulSoup(content, "html.parser")
        
        stats = {"text_chunks": 0, "tables": 0}
        
        # Extract tables
        tables = self.table_extractor.extract_html(content, file_path)
        for table in tables:
            table.embedding = self.text_embeddings.embed_text(table.text)
            self.index.add(table)
            stats["tables"] += 1
        
        # Extract text (excluding tables)
        for table in soup.find_all("table"):
            table.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        if text:
            text_stats = self.add_text(text, file_path)
            stats["text_chunks"] = text_stats["text_chunks"]
        
        self.sources[file_path] = {"type": "html", "stats": stats}
        return stats
    
    # =========================================================================
    # Querying
    # =========================================================================
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        include_images: bool = True,
    ) -> MultimodalResponse:
        """
        Query across all modalities.
        
        Args:
            question: Natural language question
            k: Number of results per modality
            include_images: Include image data in response
            
        Returns:
            MultimodalResponse with answer and context
        """
        k = k or self.config.n_results
        
        # Retrieve context
        context = self._retrieve_context(question, k)
        
        # Build context string
        context_str = self._format_context(context)
        
        # Prepare images for vision model if relevant
        images_for_llm = []
        if include_images and context.image_results:
            for result in context.image_results[:3]:  # Limit images
                if isinstance(result.item, ImageContent) and result.item.image_data:
                    images_for_llm.append(ImageInput(
                        data=result.item.image_data,
                        media_type=f"image/{result.item.format}",
                    ))
        
        # Generate answer
        prompt = self.QUERY_PROMPT.format(
            context=context_str,
            question=question,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            images=images_for_llm if images_for_llm else None,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Collect images and tables for response
        images = [
            result.item for result in context.image_results
            if isinstance(result.item, ImageContent)
        ]
        
        tables = [
            result.item for result in context.table_results
            if isinstance(result.item, TableContent)
        ]
        
        return MultimodalResponse(
            answer=response.content,
            context=context,
            query=question,
            model=response.model,
            images=images,
            tables=tables,
        )
    
    def query_stream(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Generator[str, None, MultimodalResponse]:
        """Stream a query response."""
        k = k or self.config.n_results
        
        context = self._retrieve_context(question, k)
        context_str = self._format_context(context)
        
        prompt = self.QUERY_PROMPT.format(
            context=context_str,
            question=question,
        )
        
        full_answer = ""
        for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            full_answer += chunk
            yield chunk
        
        images = [
            result.item for result in context.image_results
            if isinstance(result.item, ImageContent)
        ]
        
        tables = [
            result.item for result in context.table_results
            if isinstance(result.item, TableContent)
        ]
        
        return MultimodalResponse(
            answer=full_answer,
            context=context,
            query=question,
            model=getattr(self.llm, 'model', 'unknown'),
            images=images,
            tables=tables,
        )
    
    def _retrieve_context(self, question: str, k: int) -> MultimodalContext:
        """Retrieve relevant content across all modalities."""
        # Search each modality
        text_results = self.index.search_text(question, k=k)
        image_results = self.index.search_images(question, k=k)
        table_results = self.index.search_tables(question, k=k)
        
        return MultimodalContext(
            text_results=text_results,
            image_results=image_results,
            table_results=table_results,
        )
    
    def _format_context(self, context: MultimodalContext) -> str:
        """Format multimodal context for the prompt."""
        parts = []
        
        # Text context
        if context.text_results:
            parts.append("=== TEXT CONTENT ===")
            for i, result in enumerate(context.text_results, 1):
                parts.append(f"[{i}] Source: {result.item.source_file}")
                parts.append(result.item.text[:1000])
                parts.append("")
        
        # Image context
        if context.image_results:
            parts.append("=== IMAGES ===")
            for result in context.image_results:
                item = result.item
                if isinstance(item, ImageContent):
                    chart_info = ""
                    if isinstance(item, ChartContent):
                        chart_info = f"Chart Type: {item.chart_type}"
                        if item.data_points:
                            chart_info += f"\nData: {item.data_points[:5]}"
                    
                    parts.append(self.IMAGE_CONTEXT_TEMPLATE.format(
                        description=item.description or item.text,
                        source=item.source_file,
                        image_type=item.image_type,
                        chart_info=chart_info,
                    ))
                    parts.append("")
        
        # Table context
        if context.table_results:
            parts.append("=== TABLES ===")
            for result in context.table_results:
                item = result.item
                if isinstance(item, TableContent):
                    # Sample first few rows
                    sample_rows = item.rows[:5]
                    sample = item.to_markdown()[:500] if item.to_markdown() else ""
                    
                    parts.append(self.TABLE_CONTEXT_TEMPLATE.format(
                        caption=item.caption or "Untitled Table",
                        source=item.source_file,
                        columns=", ".join(item.headers),
                        num_rows=len(item.rows),
                        sample=sample,
                    ))
                    parts.append("")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Specialized Queries
    # =========================================================================
    
    def find_images(
        self,
        query: str,
        k: int = 5,
    ) -> list[ImageContent]:
        """Find images matching a query."""
        results = self.index.search_images(query, k=k)
        return [r.item for r in results if isinstance(r.item, ImageContent)]
    
    def find_tables(
        self,
        query: str,
        k: int = 5,
    ) -> list[TableContent]:
        """Find tables matching a query."""
        results = self.index.search_tables(query, k=k)
        return [r.item for r in results if isinstance(r.item, TableContent)]
    
    def find_charts(
        self,
        query: str,
        k: int = 5,
    ) -> list[ChartContent]:
        """Find charts matching a query."""
        results = self.index.search(
            query,
            k=k,
            content_types=[ContentType.CHART],
        )
        return [r.item for r in results if isinstance(r.item, ChartContent)]
    
    def query_table(
        self,
        table_id: str,
        column: str,
        value: str,
    ) -> list[dict]:
        """Query a specific table."""
        return self.index.query_table(table_id, column, value)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "sources": len(self.sources),
            "index": self.index.get_stats(),
            "config": {
                "llm_provider": self.config.llm_provider,
                "embedding_provider": self.config.embedding_provider,
            },
        }
    
    def list_sources(self) -> list[dict]:
        """List all indexed sources."""
        return [
            {"file": path, **info}
            for path, info in self.sources.items()
        ]
