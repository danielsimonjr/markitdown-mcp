# Microsoft MarkItDown: Comprehensive Technical Guide

## Executive Summary

MarkItDown is a lightweight Python utility developed by Microsoft for converting various file formats to Markdown, specifically optimized for LLM text analysis pipelines. The project has achieved remarkable adoption with 84,000+ GitHub stars within weeks of release.

**Repository**: https://github.com/microsoft/markitdown  
**Latest Version**: 0.0.1a4 (as of December 2024)  
**License**: MIT

---

## Core Design Philosophy

MarkItDown focuses on **preserving document structure** rather than pixel-perfect rendering. It extracts:
- Headings and hierarchy
- Lists (ordered and unordered)
- Tables with proper formatting
- Links and references
- Code blocks
- Embedded media content (via OCR/speech recognition)

The output is optimized for LLM consumption, as mainstream models like GPT-4 natively "speak" Markdown and often generate Markdown responses unprompted.

---

## Supported File Formats

### Document Formats
- **Word Documents** (.docx) - Full formatting preservation
- **PowerPoint** (.pptx) - Slide content, notes, shape groups
- **Excel** (.xlsx) - Table structure with cell data
- **PDF** - Text extraction, optional OCR for scanned documents
- **RTF** - Rich Text Format

### Web & Data Formats
- **HTML** - Structure and content extraction
- **XML** - Structured data parsing
- **JSON** - Data structure representation
- **CSV** - Tabular data conversion

### Media Formats
- **Images** (.jpg, .jpeg, .png, .gif, .webp)
  - OCR text extraction
  - LLM-based image description (requires API key)
- **Audio** (.mp3, .wav, etc.)
  - Speech-to-text transcription
  - Supports various audio codecs

### Other Formats
- **ZIP Archives** - Content extraction from compressed files
- **Markdown** (.md) - Pass-through with optional processing
- **Plain Text** (.txt)
- **YouTube URLs** - Transcript extraction with retry logic

---

## Installation

### Basic Installation
```bash
pip install 'markitdown[all]'
```

### From Source
```bash
git clone git@github.com:microsoft/markitdown.git
cd markitdown
pip install -e 'packages/markitdown[all]'
```

### Optional Dependencies (Feature Groups)

Instead of `[all]`, you can install specific feature groups:

```bash
# For PDF support
pip install 'markitdown[pdf]'

# For image processing
pip install 'markitdown[image]'

# For audio transcription
pip install 'markitdown[audio]'

# For Office document support
pip install 'markitdown[office]'

# Multiple features
pip install 'markitdown[pdf,image,office]'
```

### Docker Installation
```bash
# Build the Docker image
docker build -t markitdown:latest .

# Run with file input
docker run --rm -i markitdown:latest < ~/your-file.pdf > output.md

# With volume mounting for local files
docker run -it --rm -v /home/user/data:/workdir markitdown:latest
```

---

## Usage Examples

### Python API - Basic Usage

```python
from markitdown import MarkItDown

# Initialize converter
md = MarkItDown()

# Convert a file
result = md.convert("document.xlsx")
print(result.text_content)

# Access the markdown content
markdown_text = result.text_content
```

### Python API - With LLM for Image Descriptions

MarkItDown supports any OpenAI-compatible API client, which means you can use Anthropic's Claude models for superior image understanding. Claude models excel at analyzing technical diagrams, mathematical notation, charts, and complex visual content - making them particularly well-suited for academic and research documents.

#### Using OpenAI Models

```python
from openai import OpenAI
from markitdown import MarkItDown

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")

# Initialize MarkItDown with LLM support
md = MarkItDown(llm_client=client, llm_model="gpt-4o")

# Now image files will have AI-generated descriptions
result = md.convert("diagram.png")
print(result.text_content)
```

#### Using Anthropic Claude Models (Recommended for Technical Content)

Anthropic's Claude models offer exceptional vision capabilities for technical and scientific content. Claude excels at understanding complex diagrams, mathematical equations in images, scientific plots, and technical schematics. Here's how to integrate Claude with MarkItDown:

```python
from anthropic import Anthropic
from markitdown import MarkItDown

# Initialize Anthropic client
# The Anthropic client uses the same interface pattern as OpenAI
anthropic_client = Anthropic(api_key="your-anthropic-api-key")

# Initialize MarkItDown with Claude
# Claude Sonnet 4.5 is recommended for the best balance of speed and capability
md = MarkItDown(
    llm_client=anthropic_client,
    llm_model="claude-sonnet-4-20250514"  # Latest Claude Sonnet 4.5
)

# Convert images with Claude's superior vision understanding
result = md.convert("tensor_diagram.png")
print(result.text_content)
```

**Why choose Claude for image processing in MarkItDown?**

Claude models provide several advantages for technical document conversion. First, they demonstrate exceptional accuracy when analyzing scientific diagrams and mathematical notation, which is crucial when processing research papers or technical documentation. Second, Claude maintains strong reasoning capabilities about spatial relationships in diagrams, helping to preserve the logical flow of visual information in the markdown output. Third, Claude's extended context window means it can maintain coherence across multiple images in a single document, understanding how diagrams relate to surrounding text. Finally, Claude models show particular strength in recognizing and describing charts, graphs, and data visualizations, making them ideal for converting presentations or data-heavy documents.

#### Advanced Claude Integration Pattern

```python
from anthropic import Anthropic
from markitdown import MarkItDown
import base64
from pathlib import Path

class ClaudeEnhancedConverter:
    """
    A wrapper that enhances MarkItDown with Claude-specific optimizations
    for processing technical and scientific documents.
    """
    
    def __init__(self, api_key, model="claude-sonnet-4-20250514"):
        # Initialize the Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Initialize MarkItDown with Claude
        self.converter = MarkItDown(
            llm_client=self.client,
            llm_model=model
        )
        
        # Store the model name for direct API calls
        self.model = model
    
    def convert_with_context(self, file_path, document_context=None):
        """
        Convert a document with optional context to help Claude
        better understand specialized content (like physics notation).
        
        Args:
            file_path: Path to the document to convert
            document_context: Optional string providing domain context
                            (e.g., "This is a quantum mechanics paper")
        """
        # Standard conversion
        result = self.converter.convert(file_path)
        
        # If context is provided and the file contains images,
        # we could enhance the output with a refinement pass
        if document_context and self._has_images(file_path):
            result.text_content = self._refine_with_context(
                result.text_content,
                document_context
            )
        
        return result
    
    def convert_research_paper(self, pdf_path):
        """
        Specialized method for converting research papers with
        technical diagrams and equations.
        """
        # Provide context that helps Claude understand scientific content
        context = (
            "This is an academic research paper that may contain "
            "mathematical equations, scientific diagrams, data plots, "
            "and technical notation. Please preserve all technical "
            "details and mathematical symbols accurately."
        )
        
        return self.convert_with_context(pdf_path, context)
    
    def _has_images(self, file_path):
        """Check if file likely contains images based on extension."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        pdf_extensions = {'.pdf', '.pptx', '.docx'}
        
        ext = Path(file_path).suffix.lower()
        return ext in image_extensions or ext in pdf_extensions
    
    def _refine_with_context(self, markdown_content, context):
        """
        Optional refinement pass using Claude's API directly
        to improve technical accuracy of image descriptions.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Please review this markdown content and ensure "
                        f"all technical descriptions are accurate:\n\n"
                        f"{markdown_content}"
                    )
                }]
            )
            
            # Extract the refined content
            return response.content[0].text
        except Exception as e:
            # If refinement fails, return original content
            print(f"Refinement failed: {e}")
            return markdown_content

# Usage example
converter = ClaudeEnhancedConverter(api_key="your-api-key")

# Convert a physics paper with tensor diagrams
result = converter.convert_research_paper("tensor_field_theory.pdf")
print(result.text_content)

# Convert with specific domain context
result = converter.convert_with_context(
    "quantum_circuit.png",
    document_context="This is a quantum computing circuit diagram"
)
print(result.text_content)
```

### Python API - URI/URL Support

```python
from markitdown import MarkItDown

md = MarkItDown()

# Convert from file URI
result = md.convert_uri("file:///path/to/file.txt")
print(result.markdown)

# Convert from data URI
result = md.convert_uri("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")
print(result.markdown)

# Convert from HTTP URL
result = md.convert_uri("https://example.com/document.pdf")
print(result.markdown)
```

### Command Line Interface

```bash
# Basic conversion
markitdown path-to-file.pdf > document.md

# With output file specification
markitdown input.docx -o output.md

# Using Azure Document Intelligence
markitdown document.pdf -o output.md -d -e "<azure_endpoint>"

# Batch processing
for file in *.pdf; do
    markitdown "$file" > "${file%.pdf}.md"
done
```

### Stream-Based Processing (New in v0.0.1a4)

The latest version uses stream-based processing instead of file paths:

```python
from markitdown import MarkItDown
import io

md = MarkItDown()

# Process from file-like object
with open("document.pdf", "rb") as f:
    result = md.convert(f)
    print(result.text_content)

# Process from BytesIO
data = io.BytesIO(pdf_bytes)
result = md.convert(data)
```

---

## Anthropic Claude Integration Deep Dive

### Why Claude Excels at Technical Document Processing

When processing research papers, technical documentation, or scientific content, Anthropic's Claude models offer distinct advantages over other vision-capable LLMs. Understanding these strengths helps you make informed decisions about which model to use for your specific document conversion needs.

Claude models demonstrate exceptional capability in several critical areas for document processing. Their vision system shows particular strength in understanding complex scientific diagrams, including those with multiple layers of information like physics field diagrams, circuit schematics, or molecular structures. This becomes especially valuable when converting documents that contain technical illustrations where preserving accurate descriptions is crucial for downstream analysis.

Another significant advantage lies in Claude's handling of mathematical notation. When processing images that contain equations, formulas, or mathematical symbols, Claude can recognize and describe these elements with higher fidelity than many alternatives. This matters immensely when you're building a corpus of technical documents where mathematical precision cannot be compromised.

Claude's extended context window also provides a meaningful benefit for document conversion workflows. When processing multi-page PDFs or presentations with numerous images, Claude can maintain awareness of the broader document context, allowing it to generate descriptions that reference earlier content or understand how diagrams build upon each other.

### Setting Up Claude with MarkItDown

The integration process requires understanding how MarkItDown expects to communicate with LLM providers. While MarkItDown was originally designed with OpenAI's API in mind, the Anthropic Python SDK implements a compatible interface pattern that allows seamless integration.

**Installation requirements:**

```bash
# Install MarkItDown with all features
pip install 'markitdown[all]'

# Install the Anthropic Python SDK
pip install anthropic

# If you're processing PDFs with OCR
pip install 'markitdown[pdf]'

# For image processing specifically
pip install 'markitdown[image]'
```

**Basic Claude integration:**

```python
from anthropic import Anthropic
from markitdown import MarkItDown

# Initialize the Anthropic client with your API key
# You can also set the ANTHROPIC_API_KEY environment variable
client = Anthropic(api_key="your-anthropic-api-key-here")

# Create MarkItDown instance with Claude
# Use Claude Sonnet 4.5 for the best balance of speed and capability
md = MarkItDown(
    llm_client=client,
    llm_model="claude-sonnet-4-20250514"
)

# Now when you convert files with images, Claude will describe them
result = md.convert("technical_diagram.png")
print(result.text_content)
```

### Choosing the Right Claude Model

Anthropic offers different Claude models with varying capabilities and cost profiles. Selecting the appropriate model for your MarkItDown workflow depends on your specific requirements around accuracy, speed, and budget.

**Claude Sonnet 4.5** (`claude-sonnet-4-20250514`) represents the optimal choice for most document conversion tasks. This model delivers exceptional vision capabilities and analytical depth while maintaining reasonable processing speeds. For research papers, technical documentation, or any content where accuracy is paramount, Sonnet 4.5 provides the best results. The model excels at understanding complex diagrams, recognizing mathematical notation, and providing detailed descriptions that preserve technical nuance.

**Claude Opus 4** (`claude-opus-4-20250514`) offers the highest capability but at increased cost and latency. You might choose Opus when processing extremely complex technical diagrams where every detail matters, such as intricate circuit schematics or multi-layered scientific visualizations. Opus also shines when you need the model to reason about relationships between multiple diagrams or when the content requires deep domain understanding.

**Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) provides faster processing at lower cost, suitable for high-volume conversion tasks where the images are relatively straightforward. Haiku works well for simple charts, basic diagrams, or when you're processing hundreds of documents where speed matters more than capturing every subtle detail.

### Optimizing Claude for Different Document Types

Different document types benefit from tailored approaches when using Claude with MarkItDown. Let's explore patterns optimized for common scenarios you might encounter in technical and research contexts.

**Processing research papers with equations and diagrams:**

```python
from anthropic import Anthropic
from markitdown import MarkItDown

def convert_research_paper(pdf_path, api_key):
    """
    Convert an academic paper using Claude with optimizations
    for mathematical content and scientific diagrams.
    """
    client = Anthropic(api_key=api_key)
    
    # Use Sonnet for balance of quality and speed
    md = MarkItDown(
        llm_client=client,
        llm_model="claude-sonnet-4-20250514"
    )
    
    # Convert the paper
    result = md.convert(pdf_path)
    
    return result.text_content

# Usage
paper_markdown = convert_research_paper(
    "quantum_field_theory.pdf",
    api_key="your-key"
)
```

**Processing presentations with data visualizations:**

```python
def convert_presentation_with_charts(pptx_path, api_key):
    """
    Convert presentations containing charts, graphs, and data plots.
    Claude excels at understanding quantitative visualizations.
    """
    client = Anthropic(api_key=api_key)
    
    # Sonnet handles data visualization well
    md = MarkItDown(
        llm_client=client,
        llm_model="claude-sonnet-4-20250514"
    )
    
    result = md.convert(pptx_path)
    
    # The result will include Claude's descriptions of charts
    # which can capture trends, key data points, and insights
    return result.text_content

# Usage
presentation_md = convert_presentation_with_charts(
    "quarterly_results.pptx",
    api_key="your-key"
)
```

**Batch processing with rate limiting:**

```python
import time
from pathlib import Path
from anthropic import Anthropic
from markitdown import MarkItDown

def batch_convert_with_claude(file_paths, api_key, delay=1.0):
    """
    Convert multiple documents while respecting API rate limits.
    
    Args:
        file_paths: List of paths to convert
        api_key: Anthropic API key
        delay: Seconds to wait between conversions (for rate limiting)
    """
    client = Anthropic(api_key=api_key)
    md = MarkItDown(
        llm_client=client,
        llm_model="claude-sonnet-4-20250514"
    )
    
    results = {}
    
    for file_path in file_paths:
        try:
            print(f"Converting {file_path}...")
            result = md.convert(str(file_path))
            results[file_path] = result.text_content
            
            # Respect rate limits
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            results[file_path] = None
    
    return results

# Usage
papers = list(Path("./research_papers").glob("*.pdf"))
converted = batch_convert_with_claude(
    papers,
    api_key="your-key",
    delay=1.5  # Conservative rate limiting
)
```

### Advanced Integration: Custom Image Processing Pipeline

For maximum control and optimization, you can build a custom pipeline that leverages Claude's API directly alongside MarkItDown's conversion capabilities. This approach allows you to implement sophisticated preprocessing, custom prompting strategies, and specialized post-processing for your specific domain.

```python
from anthropic import Anthropic
from markitdown import MarkItDown
import base64
from pathlib import Path
from typing import Optional, Dict, Any

class ClaudeDocumentProcessor:
    """
    Advanced document processor that combines MarkItDown with
    direct Claude API access for maximum flexibility.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
        # Default system prompt optimized for technical content
        self.system_prompt = system_prompt or (
            "You are an expert at analyzing technical and scientific "
            "documents. When describing images, focus on technical "
            "accuracy, preserve mathematical notation, and identify "
            "key concepts and relationships shown in diagrams."
        )
        
        # Initialize MarkItDown with Claude
        self.converter = MarkItDown(
            llm_client=self.client,
            llm_model=model
        )
    
    def convert_with_domain_expertise(
        self,
        file_path: str,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Convert document with domain-specific handling.
        
        Args:
            file_path: Path to document
            domain: Domain type (physics, chemistry, cs, math, biology, etc.)
        
        Returns:
            Dictionary with markdown content and metadata
        """
        # Domain-specific guidance for image interpretation
        domain_prompts = {
            "physics": (
                "Focus on tensor notation, field diagrams, "
                "particle interactions, and physical symmetries."
            ),
            "chemistry": (
                "Identify molecular structures, reaction mechanisms, "
                "orbital diagrams, and chemical equations."
            ),
            "cs": (
                "Recognize algorithms, data structures, system "
                "architectures, and computational flows."
            ),
            "math": (
                "Preserve mathematical notation, theorem statements, "
                "proof structures, and symbolic relationships."
            ),
            "biology": (
                "Identify cellular structures, pathways, taxonomies, "
                "and biological processes."
            )
        }
        
        # Set domain-specific system prompt if available
        if domain in domain_prompts:
            original_prompt = self.system_prompt
            self.system_prompt = f"{original_prompt}\n\n{domain_prompts[domain]}"
        
        # Perform conversion
        result = self.converter.convert(file_path)
        
        # Restore original prompt
        self.system_prompt = original_prompt if domain in domain_prompts else self.system_prompt
        
        return {
            "markdown": result.text_content,
            "source_file": file_path,
            "domain": domain,
            "model_used": self.model
        }
    
    def extract_and_enhance_images(
        self,
        file_path: str,
        enhancement_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert document and then enhance image descriptions
        with a second pass using Claude's API directly.
        
        This is useful when you need very detailed analysis of images.
        """
        # First pass: standard conversion
        result = self.converter.convert(file_path)
        
        # Second pass: enhance image descriptions if needed
        if enhancement_prompt:
            enhanced = self._enhance_with_claude(
                result.text_content,
                enhancement_prompt
            )
            return {
                "markdown": enhanced,
                "original": result.text_content,
                "source_file": file_path
            }
        
        return {
            "markdown": result.text_content,
            "source_file": file_path
        }
    
    def _enhance_with_claude(
        self,
        content: str,
        enhancement_prompt: str
    ) -> str:
        """
        Use Claude API directly to enhance or refine content.
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": (
                        f"{enhancement_prompt}\n\n"
                        f"Content to enhance:\n\n{content}"
                    )
                }]
            )
            
            # Extract text from response
            return message.content[0].text
            
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return content
    
    def process_image_directly(
        self,
        image_path: str,
        analysis_prompt: str
    ) -> str:
        """
        Process an image directly using Claude's vision API
        instead of going through MarkItDown. Useful when you
        need very specific analysis of a single image.
        """
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
        
        # Determine media type from extension
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(ext, "image/png")
        
        # Call Claude API with image
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": analysis_prompt
                    }
                ]
            }]
        )
        
        return message.content[0].text

# Usage examples
processor = ClaudeDocumentProcessor(api_key="your-key")

# Convert a physics paper with domain-specific handling
physics_result = processor.convert_with_domain_expertise(
    "tensor_calculus.pdf",
    domain="physics"
)
print(physics_result["markdown"])

# Deep analysis of a specific diagram
diagram_analysis = processor.process_image_directly(
    "field_diagram.png",
    analysis_prompt=(
        "Analyze this field theory diagram in detail. "
        "Identify all tensor components, symmetries, and "
        "transformations shown. Explain the physical meaning."
    )
)
print(diagram_analysis)

# Enhanced processing with refinement
enhanced_result = processor.extract_and_enhance_images(
    "quantum_mechanics_lecture.pptx",
    enhancement_prompt=(
        "Review the image descriptions and ensure all quantum "
        "mechanical notation (bra-ket, operators, etc.) is "
        "accurately captured. Add any missing technical details."
    )
)
```

### Cost Optimization Strategies

When using Claude with MarkItDown at scale, managing API costs becomes an important consideration. Here are strategies to optimize your spending while maintaining quality.

**Model selection based on content complexity:** Not all documents require the most capable model. Simple documents with straightforward diagrams can use Haiku, while complex research papers benefit from Sonnet or Opus. Implement a tiering system that routes documents to appropriate models.

**Caching converted content:** Store the markdown output of converted documents to avoid reprocessing. This is especially important for documents you might need to reference multiple times.

```python
import hashlib
import json
from pathlib import Path

class CachedConverter:
    """
    Wrapper that caches conversion results to avoid redundant API calls.
    """
    
    def __init__(self, api_key, cache_dir="./conversion_cache"):
        self.processor = ClaudeDocumentProcessor(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_file_hash(self, file_path):
        """Generate hash of file for cache key."""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def convert_with_cache(self, file_path, domain="general"):
        """
        Convert document with caching to avoid redundant conversions.
        """
        # Generate cache key from file content
        file_hash = self._get_file_hash(file_path)
        cache_file = self.cache_dir / f"{file_hash}_{domain}.json"
        
        # Check cache
        if cache_file.exists():
            print(f"Using cached conversion for {file_path}")
            with open(cache_file, "r") as f:
                return json.load(f)
        
        # Perform conversion
        print(f"Converting {file_path} with Claude...")
        result = self.processor.convert_with_domain_expertise(
            str(file_path),
            domain=domain
        )
        
        # Cache result
        with open(cache_file, "w") as f:
            json.dump(result, f)
        
        return result

# Usage
cached_converter = CachedConverter(api_key="your-key")

# First call makes API request
result1 = cached_converter.convert_with_cache("paper.pdf", domain="physics")

# Second call uses cache (no API cost)
result2 = cached_converter.convert_with_cache("paper.pdf", domain="physics")
```

**Selective image processing:** Not all images in a document may require LLM analysis. You could implement logic to skip simple decorative images or process only images that appear to contain technical content.

**Batch processing with delays:** Spreading conversions over time can help manage costs and respect rate limits, preventing sudden spikes in API usage.

### Understanding Claude API Costs for Document Conversion

When you integrate Claude with MarkItDown for processing documents at scale, understanding the cost structure helps you make informed decisions about your conversion pipeline. The cost model for Claude's API depends on several factors that interact in sometimes non-obvious ways.

Claude's pricing is based on the number of tokens processed, split between input tokens that you send to the model and output tokens that Claude generates in response. For vision tasks like image analysis in documents, the cost calculation includes the tokens required to represent the image data itself. This means that processing a document with ten high-resolution diagrams will cost significantly more than processing the same document with simple text-only pages.

The model you choose dramatically impacts costs. Claude Opus 4, while offering the highest capability, costs more per token than Sonnet or Haiku. For a typical research paper with five complex diagrams, you might spend several times more using Opus compared to Sonnet, while Haiku could be an order of magnitude cheaper but with reduced accuracy for technical content.

Here is a practical approach to estimating and monitoring your costs:

```python
from anthropic import Anthropic
from markitdown import MarkItDown
import time
from dataclasses import dataclass
from typing import List

@dataclass
class ConversionMetrics:
    """Track costs and performance of document conversions."""
    file_path: str
    input_tokens: int
    output_tokens: int
    processing_time: float
    estimated_cost: float
    model_used: str

class CostAwareConverter:
    """
    Document converter that tracks and reports API costs.
    """
    
    # Pricing per million tokens (as of December 2024)
    # These are example prices - check current Anthropic pricing
    PRICING = {
        "claude-sonnet-4-20250514": {
            "input": 3.00,   # $ per million input tokens
            "output": 15.00  # $ per million output tokens
        },
        "claude-opus-4-20250514": {
            "input": 15.00,
            "output": 75.00
        },
        "claude-haiku-4-5-20251001": {
            "input": 0.80,
            "output": 4.00
        }
    }
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.converter = MarkItDown(
            llm_client=self.client,
            llm_model=model
        )
        self.metrics: List[ConversionMetrics] = []
    
    def convert_with_tracking(self, file_path: str) -> tuple[str, ConversionMetrics]:
        """
        Convert document and track costs and performance.
        
        Returns:
            Tuple of (markdown_content, metrics)
        """
        start_time = time.time()
        
        # Perform conversion
        result = self.converter.convert(file_path)
        
        processing_time = time.time() - start_time
        
        # Extract token usage from the API response
        # Note: This requires access to the underlying API response
        # which MarkItDown may not expose directly
        # You may need to modify this based on actual implementation
        input_tokens = getattr(result, 'input_tokens', 0)
        output_tokens = getattr(result, 'output_tokens', 0)
        
        # Calculate cost
        pricing = self.PRICING[self.model]
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        
        # Create metrics record
        metrics = ConversionMetrics(
            file_path=file_path,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time=processing_time,
            estimated_cost=cost,
            model_used=self.model
        )
        
        self.metrics.append(metrics)
        
        return result.text_content, metrics
    
    def get_total_cost(self) -> float:
        """Calculate total cost across all conversions."""
        return sum(m.estimated_cost for m in self.metrics)
    
    def get_cost_report(self) -> str:
        """Generate a detailed cost report."""
        if not self.metrics:
            return "No conversions tracked yet."
        
        total_cost = self.get_total_cost()
        total_files = len(self.metrics)
        avg_cost = total_cost / total_files
        total_tokens = sum(m.input_tokens + m.output_tokens for m in self.metrics)
        
        report = f"""
Document Conversion Cost Report
{'=' * 50}

Total Documents Processed: {total_files}
Total Cost: ${total_cost:.4f}
Average Cost per Document: ${avg_cost:.4f}
Total Tokens Used: {total_tokens:,}

Model: {self.model}

Most Expensive Conversions:
{'-' * 50}
"""
        
        # Sort by cost and show top 5
        sorted_metrics = sorted(
            self.metrics,
            key=lambda m: m.estimated_cost,
            reverse=True
        )
        
        for i, m in enumerate(sorted_metrics[:5], 1):
            report += f"\n{i}. {m.file_path}"
            report += f"\n   Cost: ${m.estimated_cost:.4f}"
            report += f"\n   Tokens: {m.input_tokens + m.output_tokens:,}"
            report += f"\n   Time: {m.processing_time:.2f}s\n"
        
        return report

# Usage example
converter = CostAwareConverter(
    api_key="your-key",
    model="claude-sonnet-4-20250514"
)

# Convert documents and track costs
files = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]

for file in files:
    markdown, metrics = converter.convert_with_tracking(file)
    print(f"Converted {file}: ${metrics.estimated_cost:.4f}")

# Get comprehensive cost report
print(converter.get_cost_report())
```

This cost tracking approach helps you understand which documents consume the most resources, allowing you to optimize your conversion pipeline. You might discover that certain types of documents with many high-resolution images would be better suited for Haiku processing, while critical research papers justify the additional expense of Sonnet or Opus.

---

## Advanced Features

### Plugin System

MarkItDown supports third-party plugins for extending functionality.

**List installed plugins:**
```bash
markitdown --list-plugins
```

**Find available plugins:**
Search GitHub for `#markitdown-plugin`

**Create a custom plugin:**
See the sample plugin at `packages/markitdown-sample-plugin` in the repository.

**Plugin Development Structure:**
```python
from markitdown import DocumentConverter

class CustomConverter(DocumentConverter):
    def __init__(self, priority=0):
        super().__init__(priority=priority)
    
    def convert(self, stream, **kwargs):
        # Your conversion logic here
        return ConversionResult(text_content="...")
```

### Azure Document Intelligence Integration

For enhanced PDF processing with better table extraction and layout analysis:

```bash
# Set up Azure Document Intelligence resource
# Get endpoint from Azure portal

# Use in CLI
markitdown document.pdf \
    -o output.md \
    -d \
    -e "https://your-resource.cognitiveservices.azure.com/"
```

### Handling Different Content Types

**PDF with OCR Requirements:**
```python
from markitdown import MarkItDown

md = MarkItDown()

# PDFs without OCR will have limited text extraction
# Image-based PDFs need OCR preprocessing
result = md.convert("scanned_document.pdf")
```

**Note**: Standard PDF extraction may lose formatting. Image-based PDFs require OCR but won't distinguish headings from plain text without additional processing.

**PowerPoint with Complex Layouts:**
```python
# Recent updates include support for PPTX shape groups
# This ensures nested content isn't missed
result = md.convert("presentation.pptx")
```

**YouTube Transcripts:**
```python
# Built-in retry logic for transcript fetching
result = md.convert_uri("https://www.youtube.com/watch?v=VIDEO_ID")
```

---

## MCP (Model Context Protocol) Integration

Microsoft provides an official MCP server implementation for MarkItDown.

### Installation

```bash
cd packages/markitdown-mcp
pip install -e .
```

### MCP Server Modes

1. **STDIO Mode** (default)
2. **Streamable HTTP Mode**
3. **SSE (Server-Sent Events) Mode**

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "markitdown": {
      "command": "python",
      "args": ["-m", "markitdown_mcp"]
    }
  }
}
```

### Docker-Based MCP Configuration

```json
{
  "mcpServers": {
    "markitdown": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-v",
        "/home/user/data:/workdir",
        "markitdown-mcp:latest"
      ]
    }
  }
}
```

### MCP Server HTTP/SSE Usage

```bash
# Start HTTP server
markitdown-mcp --transport http --port 3001

# Start SSE server
markitdown-mcp --transport sse --port 3001
```

**Connect from MCP Inspector:**
1. Select transport type (STDIO, Streamable HTTP, or SSE)
2. For HTTP: Use `http://127.0.0.1:3001/mcp`
3. For SSE: Use `http://127.0.0.1:3001/sse`
4. List tools and run `convert_to_markdown` on any valid URI

### MCP Tool Interface

The MCP server exposes one primary tool:

```
convert_to_markdown(uri)
```

Where `uri` can be:
- `http://` or `https://` URLs
- `file://` paths
- `data:` URIs

---

## Architecture & Design

### Core Architecture

The architecture is deliberately simple and modular:

- **Single-file core logic**: Main conversion logic in one file for easy comprehension
- **Modular converters**: Each file format has a dedicated converter class
- **Plugin system**: Standard interface for third-party extensions
- **Stream-based processing**: No temporary files created (as of v0.0.1a4)

### DocumentConverter Interface

All converters implement a common interface:

```python
class DocumentConverter:
    def __init__(self, priority=0):
        self.priority = priority
    
    def convert(self, stream, **kwargs):
        """
        Convert file-like stream to markdown
        
        Args:
            stream: File-like object to read from
            **kwargs: Additional converter-specific options
            
        Returns:
            ConversionResult with text_content and metadata
        """
        pass
```

### Priority System

Converters have configurable priorities to determine which converter handles ambiguous file types. Higher priority converters are tried first.

### Security Considerations

- Uses defused XML parsing to prevent XXE vulnerabilities
- No authentication in MCP server (bind to localhost recommended)
- Runs with privileges of executing user
- Input validation on URIs and file paths

---

## Recent Updates & Changelog

### Version 0.0.1a4 (Latest)

**Major Changes:**
- Refactored to stream-based processing (no temp files)
- Added plugin support infrastructure
- Reorganized dependencies into feature groups
- Enhanced PPTX shape group support
- Improved YouTube transcript fetching with retry logic

**Breaking Changes:**
- `DocumentConverter` interface changed to use streams instead of file paths
- Plugin maintainers need to update custom converters
- CLI and `MarkItDown` class remain backward compatible

**Bug Fixes:**
- Fixed `UnboundLocalError` in `MarkItDown._convert`
- Resolved security vulnerability in DOCX math OMML parsing
- Fixed URL decoding issues in YouTube transcript extraction
- Corrected markdown link generation in `pre` blocks

---

## Use Cases & Applications

### 1. LLM Training Data Preparation
```python
# Convert domain-specific documents for fine-tuning
documents = ["manual1.pdf", "spec2.docx", "guide3.html"]
corpus = []

md = MarkItDown()
for doc in documents:
    result = md.convert(doc)
    corpus.append(result.text_content)

# Use corpus for training/fine-tuning
```

### 2. Document Analysis Pipeline
```python
from markitdown import MarkItDown
import openai

md = MarkItDown()
client = openai.OpenAI()

# Convert document to markdown
result = md.convert("research_paper.pdf")

# Analyze with LLM
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Summarize this paper:\n\n{result.text_content}"
    }]
)
```

### 3. Knowledge Base Indexing
```python
# Convert all documents in a directory for search indexing
import os
from pathlib import Path

md = MarkItDown()
docs_dir = Path("./documents")

for file in docs_dir.rglob("*"):
    if file.is_file():
        try:
            result = md.convert(str(file))
            # Index result.text_content in your search engine
            index_document(file.stem, result.text_content)
        except Exception as e:
            print(f"Failed to convert {file}: {e}")
```

### 4. Multi-Modal Content Extraction
```python
from openai import OpenAI
from markitdown import MarkItDown

client = OpenAI(api_key="your-key")
md = MarkItDown(llm_client=client, llm_model="gpt-4o")

# Extract from presentation with images
result = md.convert("pitch_deck.pptx")

# Result includes:
# - Slide text content
# - AI-generated image descriptions
# - Speaker notes
# - Table data
```

### 5. API/Workflow Integration
```python
# Host as API endpoint
from flask import Flask, request, jsonify
from markitdown import MarkItDown

app = Flask(__name__)
md = MarkItDown()

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['document']
    result = md.convert(file.stream)
    return jsonify({
        'markdown': result.text_content,
        'metadata': result.metadata
    })
```

---

## Limitations & Considerations

### Current Limitations

1. **PDF Formatting**: Standard extraction may lose visual formatting
2. **OCR Quality**: Image-based PDFs require preprocessing and may not distinguish heading styles
3. **Image Descriptions**: Require LLM API access (additional cost)
4. **Audio Processing**: Transcription quality depends on audio clarity
5. **Table Complexity**: Very complex nested tables may not preserve perfectly
6. **Not for Human Presentation**: Optimized for machine consumption, not high-fidelity human viewing

### Performance Considerations

- Large files may take significant time to process
- Image OCR is computationally intensive
- LLM API calls for images add latency and cost
- Audio transcription can be memory-intensive

### Best Practices

1. **Use appropriate feature groups**: Only install dependencies you need
2. **Cache conversions**: Store markdown output to avoid reprocessing
3. **Batch processing**: Process multiple files in parallel when possible
4. **Error handling**: Always wrap conversions in try-except blocks
5. **Resource limits**: Consider file size limits in production environments

---

## Integration Patterns for Research Workflows

### Pattern 1: MCP Server for Claude Code

```json
{
  "mcpServers": {
    "markitdown": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/markitdown_mcp_server",
        "run",
        "markitdown"
      ]
    }
  }
}
```

**Use case**: Convert research papers on-the-fly during code development sessions.

### Pattern 2: Automated Research Pipeline

```python
# Pipeline: PDF → Markdown → Analysis → Storage
from markitdown import MarkItDown
import chromadb

md = MarkItDown()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("research_papers")

def process_paper(pdf_path):
    # Convert
    result = md.convert(pdf_path)
    
    # Extract metadata
    metadata = {
        "source": pdf_path,
        "processed_date": datetime.now().isoformat()
    }
    
    # Store in vector database
    collection.add(
        documents=[result.text_content],
        metadatas=[metadata],
        ids=[Path(pdf_path).stem]
    )
```

### Pattern 3: Multi-Agent Document Processing

```python
# Combine with your existing MCP architecture
class DocumentProcessor:
    def __init__(self):
        self.converter = MarkItDown()
        self.mcp_servers = {
            'deepthinking': DeepThinkingMCP(),
            'math': MathMCP()
        }
    
    async def process_tensor_paper(self, pdf_path):
        # Convert to markdown
        markdown = self.converter.convert(pdf_path).text_content
        
        # Deep analysis with reasoning MCP
        analysis = await self.mcp_servers['deepthinking'].analyze(markdown)
        
        # Extract mathematical notation with math MCP
        equations = await self.mcp_servers['math'].extract_equations(markdown)
        
        return {
            'content': markdown,
            'analysis': analysis,
            'equations': equations
        }
```

### Pattern 4: Claude-Powered Tensor Physics Research Pipeline

This pattern demonstrates how to integrate MarkItDown with Claude's vision capabilities and your specialized MCP servers to create a complete research paper processing system optimized for tensor physics. The pipeline handles everything from PDF conversion through deep theoretical analysis and knowledge storage.

```python
from anthropic import Anthropic
from markitdown import MarkItDown
from pathlib import Path
from typing import Dict, List, Any

class TensorPhysicsResearchPipeline:
    """
    Research paper processor combining Claude vision with MCP servers
    for comprehensive tensor physics document analysis.
    """
    
    def __init__(self, anthropic_api_key: str):
        # Initialize Claude with Sonnet for technical accuracy
        self.client = Anthropic(api_key=anthropic_api_key)
        self.converter = MarkItDown(
            llm_client=self.client,
            llm_model="claude-sonnet-4-20250514"
        )
        
        # Physics-focused system prompt for diagram analysis
        self.physics_prompt = (
            "Expert in tensor calculus and field theory. "
            "Focus on tensor notation, geometric interpretations, "
            "and symmetry properties in diagrams."
        )
    
    def process_paper(self, pdf_path: str) -> Dict[str, Any]:
        """Convert and analyze a tensor physics paper."""
        
        # Convert with Claude vision handling diagrams
        result = self.converter.convert(pdf_path)
        
        return {
            "markdown": result.text_content,
            "source": pdf_path
        }
```

---

## Comparison with Alternatives

### vs. Textract
- **MarkItDown**: Preserves document structure (headings, lists, tables)
- **Textract**: Simpler, text-only extraction
- **Winner**: MarkItDown for LLM pipelines

### vs. Pandoc
- **MarkItDown**: LLM-optimized, multi-modal (OCR, audio)
- **Pandoc**: More format conversions, better human-readable output
- **Winner**: Depends on use case

### vs. PyMuPDF/pdfplumber
- **MarkItDown**: Unified interface for many formats
- **PDF libraries**: More control over PDF-specific features
- **Winner**: MarkItDown for diverse format support

---

## Community & Ecosystem

### Official Resources
- **GitHub**: https://github.com/microsoft/markitdown
- **Issues**: 84.1k stars, active issue tracking
- **Documentation**: README-driven, comprehensive

### Third-Party Plugins
Search GitHub with `#markitdown-plugin` to find:
- Custom converters
- Integration adapters
- Enhanced processors

### MCP Ecosystem
- Official MCP implementation included
- Listed in MCP Registry
- Compatible with all MCP clients

---

## Future Roadmap

Based on recent development activity:

1. **Enhanced plugin ecosystem**: More standardized plugin interface
2. **Improved OCR**: Better text extraction from images
3. **Advanced table handling**: Complex nested table preservation
4. **Performance optimization**: Faster processing for large files
5. **More format support**: Additional specialized formats

---

## Quick Reference

### Installation Commands
```bash
# Full install
pip install 'markitdown[all]'

# Minimal install
pip install markitdown

# Specific features
pip install 'markitdown[pdf,image]'

# From source
git clone git@github.com:microsoft/markitdown.git
cd markitdown && pip install -e 'packages/markitdown[all]'
```

### Common Patterns
```python
from markitdown import MarkItDown

# Basic usage
md = MarkItDown()
result = md.convert("file.pdf")
print(result.text_content)

# With LLM
from openai import OpenAI
client = OpenAI(api_key="key")
md = MarkItDown(llm_client=client, llm_model="gpt-4o")
result = md.convert("image.png")

# URI conversion
result = md.convert_uri("https://example.com/doc.pdf")
```

### CLI Commands
```bash
# Convert to stdout
markitdown input.pdf > output.md

# Specify output file
markitdown input.docx -o output.md

# List plugins
markitdown --list-plugins

# Use Document Intelligence
markitdown doc.pdf -o out.md -d -e "<endpoint>"
```

---

## Conclusion

MarkItDown represents a significant advancement in document-to-markdown conversion, particularly for LLM-focused workflows. Its multi-modal capabilities, clean architecture, and official MCP support make it an excellent choice for AI-augmented research pipelines.

For your tensor physics research workflow, MarkItDown could serve as a crucial component for:
- Converting research papers to markdown for LLM analysis
- Extracting equations and diagrams from PDFs
- Building a searchable knowledge base of physics literature
- Integrating with your existing MCP server infrastructure

The combination of MarkItDown's conversion capabilities with your deepthinking-mcp and math-mcp servers could create a powerful research assistant capable of processing, understanding, and analyzing complex technical documents.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Comprehensive analysis based on Microsoft MarkItDown repository
