"""
Image Processor
 
Processes images using vision models to:
- Generate descriptions
- Classify image types (photo, chart, diagram, etc.)
- Extract text (OCR)
- Parse charts and extract data
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import io
import re
import json

from .content_types import (
    ImageContent,
    ChartContent,
    ContentType,
    generate_content_id,
)
from ..providers.llm import BaseLLM, ImageInput


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""
    
    # Description generation
    generate_descriptions: bool = True
    description_detail: str = "detailed"  # "brief", "detailed", "comprehensive"
    
    # Classification
    classify_images: bool = True
    
    # Chart processing
    extract_chart_data: bool = True
    
    # OCR
    extract_text: bool = True


class ImageProcessor:
    """
    Processes images using vision models.
    
    Capabilities:
    - Generate detailed descriptions for semantic search
    - Classify images (photo, chart, diagram, logo, etc.)
    - Extract data from charts
    - OCR for text in images
    """
    
    DESCRIPTION_PROMPTS = {
        "brief": "Describe this image in 1-2 sentences.",
        "detailed": """Describe this image in detail. Include:
- Main subjects and objects
- Any visible text
- Colors and visual style
- If it's a chart/graph: type, axes, trends
- If it's a diagram: components and relationships
- Overall context and purpose""",
        "comprehensive": """Provide a comprehensive description of this image:

1. OVERVIEW: What type of image is this? (photo, chart, diagram, screenshot, etc.)

2. MAIN CONTENT: Describe the primary subjects, objects, or data shown.

3. TEXT: List any visible text, labels, or captions.

4. VISUAL ELEMENTS: Colors, layout, style.

5. DATA (if applicable): For charts/graphs, describe:
   - Chart type
   - Axes and labels
   - Data trends and key values
   - Legend items

6. CONTEXT: What is the purpose or meaning of this image?

7. SEARCHABLE KEYWORDS: List 5-10 keywords someone might use to find this image.""",
    }
    
    CLASSIFICATION_PROMPT = """Classify this image into one of these categories:
- photo: A photograph of real-world scene or objects
- chart: A data visualization (bar chart, line graph, pie chart, etc.)
- diagram: A schematic or flow diagram
- screenshot: A screenshot of software or website
- logo: A logo or brand image
- document: A scanned document or form
- illustration: An illustration or artwork
- table: An image of a table with data
- other: None of the above

Return ONLY the category name, nothing else."""

    CHART_EXTRACTION_PROMPT = """Analyze this chart/graph and extract the data.

Return a JSON object with:
{
    "chart_type": "bar|line|pie|scatter|area|other",
    "title": "chart title if visible",
    "x_label": "x-axis label",
    "y_label": "y-axis label",
    "legend": ["list of legend items"],
    "data_points": [
        {"label": "category/x-value", "value": number, "series": "series name if applicable"}
    ],
    "trends": "description of main trends or insights",
    "notes": "any additional observations"
}

Be as accurate as possible with the numbers. If exact values aren't clear, provide estimates.
Return ONLY the JSON, no other text."""

    OCR_PROMPT = """Extract all visible text from this image.

Format the text to preserve the original layout as much as possible.
Include:
- Headings and titles
- Body text
- Labels and captions
- Any text in tables or charts

Return only the extracted text, nothing else."""

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[ImageProcessingConfig] = None,
    ):
        """
        Initialize the image processor.
        
        Args:
            llm: Vision-capable LLM
            config: Processing configuration
        """
        self.llm = llm
        self.config = config or ImageProcessingConfig()
        
        if not llm.supports_vision:
            raise ValueError("LLM must support vision for image processing")
    
    def process_image(
        self,
        image_data: bytes,
        source_file: str = "unknown",
        media_type: str = "image/jpeg",
    ) -> ImageContent:
        """
        Process a single image.
        
        Args:
            image_data: Raw image bytes
            source_file: Source file path
            media_type: MIME type
            
        Returns:
            Processed ImageContent
        """
        image_input = ImageInput(data=image_data, media_type=media_type)
        
        # Get image dimensions
        width, height = self._get_image_dimensions(image_data)
        
        # Classify image
        image_type = "unknown"
        if self.config.classify_images:
            image_type = self._classify_image(image_input)
        
        # Generate description
        description = None
        if self.config.generate_descriptions:
            description = self._describe_image(image_input)
        
        # Check if it's a chart
        if image_type == "chart" and self.config.extract_chart_data:
            return self._process_chart(
                image_data=image_data,
                image_input=image_input,
                source_file=source_file,
                media_type=media_type,
                description=description,
                width=width,
                height=height,
            )
        
        # Extract text if enabled
        ocr_text = None
        if self.config.extract_text:
            ocr_text = self._extract_text(image_input)
        
        # Build text representation
        text_parts = []
        if description:
            text_parts.append(description)
        if ocr_text:
            text_parts.append(f"Text in image: {ocr_text}")
        
        return ImageContent(
            id=generate_content_id(source_file, ContentType.IMAGE, hash(image_data)),
            content_type=ContentType.IMAGE,
            source_file=source_file,
            text="\n\n".join(text_parts) if text_parts else f"Image ({image_type})",
            image_data=image_data,
            width=width,
            height=height,
            format=media_type.split("/")[-1],
            description=description,
            image_type=image_type,
            metadata={"ocr_text": ocr_text} if ocr_text else {},
        )
    
    def process_image_file(self, file_path: str) -> ImageContent:
        """Process an image file."""
        path = Path(file_path)
        
        with open(path, "rb") as f:
            image_data = f.read()
        
        # Determine media type
        ext = path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_types.get(ext, "image/jpeg")
        
        return self.process_image(image_data, str(path), media_type)
    
    def _get_image_dimensions(self, image_data: bytes) -> tuple[int, int]:
        """Get image dimensions."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_data))
            return img.size
        except ImportError:
            return 0, 0
    
    def _classify_image(self, image_input: ImageInput) -> str:
        """Classify image type using vision model."""
        try:
            response = self.llm.generate(
                prompt=self.CLASSIFICATION_PROMPT,
                images=[image_input],
                max_tokens=50,
                temperature=0,
            )
            
            category = response.content.strip().lower()
            
            valid_categories = {
                "photo", "chart", "diagram", "screenshot",
                "logo", "document", "illustration", "table", "other"
            }
            
            return category if category in valid_categories else "other"
            
        except Exception as e:
            print(f"Warning: Image classification failed: {e}")
            return "unknown"
    
    def _describe_image(self, image_input: ImageInput) -> str:
        """Generate image description using vision model."""
        prompt = self.DESCRIPTION_PROMPTS.get(
            self.config.description_detail,
            self.DESCRIPTION_PROMPTS["detailed"],
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                images=[image_input],
                max_tokens=1000,
                temperature=0.1,
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Warning: Image description failed: {e}")
            return ""
    
    def _extract_text(self, image_input: ImageInput) -> Optional[str]:
        """Extract text from image using vision model."""
        try:
            response = self.llm.generate(
                prompt=self.OCR_PROMPT,
                images=[image_input],
                max_tokens=2000,
                temperature=0,
            )
            
            text = response.content.strip()
            return text if text and text.lower() != "no text visible" else None
            
        except Exception as e:
            print(f"Warning: OCR failed: {e}")
            return None
    
    def _process_chart(
        self,
        image_data: bytes,
        image_input: ImageInput,
        source_file: str,
        media_type: str,
        description: Optional[str],
        width: int,
        height: int,
    ) -> ChartContent:
        """Process a chart image and extract data."""
        chart_data = self._extract_chart_data(image_input)
        
        # Build text representation
        text_parts = []
        
        if chart_data.get("title"):
            text_parts.append(f"Chart: {chart_data['title']}")
        
        text_parts.append(f"Type: {chart_data.get('chart_type', 'unknown')}")
        
        if chart_data.get("x_label"):
            text_parts.append(f"X-axis: {chart_data['x_label']}")
        if chart_data.get("y_label"):
            text_parts.append(f"Y-axis: {chart_data['y_label']}")
        
        if chart_data.get("trends"):
            text_parts.append(f"Trends: {chart_data['trends']}")
        
        if description:
            text_parts.append(description)
        
        return ChartContent(
            id=generate_content_id(source_file, ContentType.CHART, hash(image_data)),
            content_type=ContentType.CHART,
            source_file=source_file,
            text="\n".join(text_parts),
            image_data=image_data,
            width=width,
            height=height,
            format=media_type.split("/")[-1],
            description=description,
            image_type="chart",
            chart_type=chart_data.get("chart_type", "unknown"),
            data_points=chart_data.get("data_points", []),
            x_label=chart_data.get("x_label"),
            y_label=chart_data.get("y_label"),
            legend=chart_data.get("legend", []),
            metadata={"raw_extraction": chart_data},
        )
    
    def _extract_chart_data(self, image_input: ImageInput) -> dict:
        """Extract structured data from a chart."""
        try:
            response = self.llm.generate(
                prompt=self.CHART_EXTRACTION_PROMPT,
                images=[image_input],
                max_tokens=2000,
                temperature=0,
            )
            
            # Parse JSON from response
            content = response.content.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            
            return {}
            
        except Exception as e:
            print(f"Warning: Chart data extraction failed: {e}")
            return {}
    
    def batch_process(
        self,
        images: list[tuple[bytes, str, str]],  # (data, source_file, media_type)
    ) -> list[ImageContent]:
        """Process multiple images."""
        results = []
        
        for image_data, source_file, media_type in images:
            try:
                result = self.process_image(image_data, source_file, media_type)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to process image from {source_file}: {e}")
        
        return results


def process_image(
    image_path: str,
    llm: BaseLLM,
    config: Optional[ImageProcessingConfig] = None,
) -> ImageContent:
    """Process a single image file."""
    processor = ImageProcessor(llm, config)
    return processor.process_image_file(image_path)
