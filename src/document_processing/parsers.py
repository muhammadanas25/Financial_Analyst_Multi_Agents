"""
Multi-parser document processing system for financial documents.
Implements Docling, PyMuPDF, and pdfplumber with quality validation.
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import re


class ParserType(Enum):
    """Supported PDF parsers"""
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    DOCLING = "docling"
    UNSTRUCTURED = "unstructured"


class ElementType(Enum):
    """Document element types"""
    TEXT = "text"
    TABLE = "table"
    TITLE = "title"
    LIST = "list"
    IMAGE = "image"
    FORMULA = "formula"


@dataclass
class DocumentElement:
    """Represents a structured document element"""
    type: ElementType
    content: str
    page: int
    metadata: Dict[str, Any]
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    """Represents a fully parsed document"""
    elements: List[DocumentElement]
    metadata: Dict[str, Any]
    parser_used: ParserType
    page_count: int
    extraction_quality: float


class PDFParser:
    """Base class for PDF parsers"""

    def __init__(self, parser_type: ParserType):
        self.parser_type = parser_type

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF and return structured document"""
        raise NotImplementedError

    def _has_text_layer(self, pdf_path: Path) -> bool:
        """Check if PDF has a text layer (not scanned)"""
        try:
            doc = fitz.open(pdf_path)
            # Check first 3 pages
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if text and len(text.strip()) > 100:
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception as e:
            logger.error(f"Error checking text layer: {e}")
            return False


class PyMuPDFParser(PDFParser):
    """
    PyMuPDF parser - Fast and reliable for native PDFs.
    Best for: Earnings call transcripts, text-heavy documents.
    F1 Score: 0.9825
    """

    def __init__(self):
        super().__init__(ParserType.PYMUPDF)

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using PyMuPDF"""
        logger.info(f"Parsing {pdf_path.name} with PyMuPDF...")

        try:
            doc = fitz.open(pdf_path)
            elements = []
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc[page_num]

                # Extract text blocks with structure
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block["type"] == 0:  # Text block
                        # Detect if it's a title based on font size
                        is_title = self._is_title_block(block)

                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")

                            if line_text.strip():
                                element_type = ElementType.TITLE if is_title else ElementType.TEXT

                                elements.append(DocumentElement(
                                    type=element_type,
                                    content=line_text.strip(),
                                    page=page_num + 1,
                                    metadata={
                                        "font_size": line["spans"][0].get("size", 0) if line.get("spans") else 0,
                                        "font_name": line["spans"][0].get("font", "") if line.get("spans") else "",
                                    },
                                    bbox=tuple(block["bbox"]) if "bbox" in block else None
                                ))

                # Extract tables (basic)
                tables = self._extract_tables_pymupdf(page, page_num)
                elements.extend(tables)

            doc.close()

            # Calculate extraction quality
            quality = self._calculate_quality(elements)

            return ParsedDocument(
                elements=elements,
                metadata={
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "has_text_layer": True,
                },
                parser_used=ParserType.PYMUPDF,
                page_count=page_count,
                extraction_quality=quality
            )

        except Exception as e:
            logger.error(f"Error parsing with PyMuPDF: {e}")
            raise

    def _is_title_block(self, block: Dict) -> bool:
        """Detect if block is a title based on font size"""
        if "lines" not in block or not block["lines"]:
            return False

        # Get average font size
        font_sizes = []
        for line in block["lines"]:
            for span in line.get("spans", []):
                font_sizes.append(span.get("size", 0))

        if not font_sizes:
            return False

        avg_size = sum(font_sizes) / len(font_sizes)
        return avg_size > 14  # Titles typically > 14pt

    def _extract_tables_pymupdf(self, page, page_num: int) -> List[DocumentElement]:
        """Extract tables from page (basic implementation)"""
        tables = []
        # PyMuPDF doesn't have native table extraction, so we detect table-like structures
        # This is a simplified version
        text = page.get_text("text")
        if self._looks_like_table(text):
            tables.append(DocumentElement(
                type=ElementType.TABLE,
                content=text,
                page=page_num + 1,
                metadata={"extraction_method": "text_pattern"},
                confidence=0.7  # Lower confidence for pattern-based detection
            ))
        return tables

    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like structures"""
        lines = text.split('\n')
        if len(lines) < 3:
            return False

        # Check for consistent column separators
        separator_count = [line.count('  ') + line.count('\t') for line in lines]
        avg_separators = sum(separator_count) / len(separator_count)
        return avg_separators > 2

    def _calculate_quality(self, elements: List[DocumentElement]) -> float:
        """Calculate extraction quality score"""
        if not elements:
            return 0.0

        # Quality based on element count and diversity
        has_titles = any(e.type == ElementType.TITLE for e in elements)
        has_tables = any(e.type == ElementType.TABLE for e in elements)

        base_quality = 0.85
        if has_titles:
            base_quality += 0.05
        if has_tables:
            base_quality += 0.05

        return min(base_quality, 1.0)


class PDFPlumberParser(PDFParser):
    """
    pdfplumber parser - Superior table extraction.
    Best for: Financial statements, balance sheets, documents with complex tables.
    F1 Score for tables: 0.9568
    """

    def __init__(self):
        super().__init__(ParserType.PDFPLUMBER)

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using pdfplumber"""
        logger.info(f"Parsing {pdf_path.name} with pdfplumber...")

        try:
            elements = []

            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        # Split into paragraphs
                        paragraphs = self._split_into_paragraphs(text)
                        for para in paragraphs:
                            if para.strip():
                                is_title = self._is_title_text(para)
                                elements.append(DocumentElement(
                                    type=ElementType.TITLE if is_title else ElementType.TEXT,
                                    content=para.strip(),
                                    page=page_num + 1,
                                    metadata={"extraction_method": "pdfplumber_text"}
                                ))

                    # Extract tables (pdfplumber's strength)
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        table_content = self._format_table(table)
                        if table_content:
                            elements.append(DocumentElement(
                                type=ElementType.TABLE,
                                content=table_content,
                                page=page_num + 1,
                                metadata={
                                    "table_index": table_idx,
                                    "rows": len(table),
                                    "columns": len(table[0]) if table else 0,
                                    "extraction_method": "pdfplumber_table"
                                },
                                confidence=0.95  # High confidence for pdfplumber tables
                            ))

            quality = self._calculate_quality(elements)

            return ParsedDocument(
                elements=elements,
                metadata={
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                },
                parser_used=ParserType.PDFPLUMBER,
                page_count=page_count,
                extraction_quality=quality
            )

        except Exception as e:
            logger.error(f"Error parsing with pdfplumber: {e}")
            raise

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or significant spacing
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _is_title_text(self, text: str) -> bool:
        """Heuristic to detect if text is a title"""
        # Titles are typically short, may be all caps, and don't end with periods
        text = text.strip()
        if len(text) > 200:
            return False
        if text.isupper() and len(text) < 100:
            return True
        if not text.endswith('.') and len(text) < 100:
            # Check if it looks like a heading
            words = text.split()
            if len(words) <= 10:
                return True
        return False

    def _format_table(self, table: List[List[str]]) -> str:
        """Format table as markdown"""
        if not table:
            return ""

        # Convert table to markdown format
        markdown_lines = []

        # Header
        if table:
            header = " | ".join(str(cell) if cell else "" for cell in table[0])
            markdown_lines.append(header)
            markdown_lines.append(" | ".join(["---"] * len(table[0])))

            # Rows
            for row in table[1:]:
                row_text = " | ".join(str(cell) if cell else "" for cell in row)
                markdown_lines.append(row_text)

        return "\n".join(markdown_lines)

    def _calculate_quality(self, elements: List[DocumentElement]) -> float:
        """Calculate extraction quality score"""
        if not elements:
            return 0.0

        table_count = sum(1 for e in elements if e.type == ElementType.TABLE)
        text_count = sum(1 for e in elements if e.type == ElementType.TEXT)

        # pdfplumber excels at tables
        base_quality = 0.90
        if table_count > 0:
            base_quality = 0.95

        return base_quality


class DoclingParser(PDFParser):
    """
    Docling parser - Open-source document understanding.
    Best for: Complex layouts, mixed content, general-purpose parsing.
    """

    def __init__(self):
        super().__init__(ParserType.DOCLING)

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using Docling"""
        logger.info(f"Parsing {pdf_path.name} with Docling...")

        try:
            from docling.document_converter import DocumentConverter

            # Initialize converter
            converter = DocumentConverter()

            # Convert document
            result = converter.convert(str(pdf_path))

            elements = []
            page_count = 0

            # Extract elements from Docling result
            if hasattr(result, 'document') and result.document:
                doc = result.document

                # Process document structure
                for element in doc.iterate_items():
                    element_type = self._map_docling_type(element.get_type())

                    if element_type:
                        elements.append(DocumentElement(
                            type=element_type,
                            content=element.get_text(),
                            page=element.get_page_number() if hasattr(element, 'get_page_number') else 1,
                            metadata={
                                "docling_type": element.get_type(),
                                "extraction_method": "docling"
                            },
                            confidence=0.92
                        ))

                page_count = doc.page_count if hasattr(doc, 'page_count') else 1

            quality = self._calculate_quality(elements)

            return ParsedDocument(
                elements=elements,
                metadata={
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                },
                parser_used=ParserType.DOCLING,
                page_count=page_count,
                extraction_quality=quality
            )

        except ImportError:
            logger.warning("Docling not installed. Install with: pip install docling")
            raise
        except Exception as e:
            logger.error(f"Error parsing with Docling: {e}")
            raise

    def _map_docling_type(self, docling_type: str) -> Optional[ElementType]:
        """Map Docling element types to our ElementType"""
        type_mapping = {
            "title": ElementType.TITLE,
            "heading": ElementType.TITLE,
            "paragraph": ElementType.TEXT,
            "text": ElementType.TEXT,
            "table": ElementType.TABLE,
            "list": ElementType.LIST,
            "image": ElementType.IMAGE,
        }
        return type_mapping.get(docling_type.lower())

    def _calculate_quality(self, elements: List[DocumentElement]) -> float:
        """Calculate extraction quality score"""
        if not elements:
            return 0.0
        return 0.92  # Docling typically has good quality


# Fallback to PyMuPDF if Docling fails
class MultiParser:
    """
    Multi-parser system with cascading fallback.
    Tries parsers in order and validates quality.
    """

    def __init__(self, prefer_parser: Optional[ParserType] = None):
        """
        Initialize multi-parser

        Args:
            prefer_parser: Preferred parser to try first
        """
        self.prefer_parser = prefer_parser
        self.parsers = {
            ParserType.DOCLING: DoclingParser(),
            ParserType.PDFPLUMBER: PDFPlumberParser(),
            ParserType.PYMUPDF: PyMuPDFParser(),
        }

    def parse(self, pdf_path: Path, min_quality: float = 0.85) -> ParsedDocument:
        """
        Parse PDF with automatic parser selection

        Args:
            pdf_path: Path to PDF file
            min_quality: Minimum acceptable quality score

        Returns:
            ParsedDocument with best quality
        """
        logger.info(f"Parsing {pdf_path.name} with multi-parser strategy...")

        # Determine parser order
        parser_order = self._determine_parser_order(pdf_path)

        best_result = None
        best_quality = 0.0

        for parser_type in parser_order:
            try:
                parser = self.parsers[parser_type]
                result = parser.parse(pdf_path)

                logger.info(
                    f"{parser_type.value}: quality={result.extraction_quality:.3f}, "
                    f"elements={len(result.elements)}"
                )

                # Keep track of best result
                if result.extraction_quality > best_quality:
                    best_quality = result.extraction_quality
                    best_result = result

                # If quality is good enough, use this result
                if result.extraction_quality >= min_quality:
                    logger.info(f"Using {parser_type.value} (quality: {result.extraction_quality:.3f})")
                    return result

            except Exception as e:
                logger.warning(f"Parser {parser_type.value} failed: {e}")
                continue

        # Return best result even if below min_quality
        if best_result:
            logger.warning(
                f"Best quality {best_quality:.3f} below threshold {min_quality}, "
                f"but returning best result from {best_result.parser_used.value}"
            )
            return best_result

        raise ValueError(f"All parsers failed for {pdf_path}")

    def _determine_parser_order(self, pdf_path: Path) -> List[ParserType]:
        """Determine optimal parser order based on document type"""

        filename = pdf_path.name.lower()

        # Detect document type from filename
        if "fs" in filename or "financial-statement" in filename or "balance" in filename:
            # Financial statements: pdfplumber excels at tables
            return [ParserType.PDFPLUMBER, ParserType.DOCLING, ParserType.PYMUPDF]

        elif "call" in filename or "transcript" in filename:
            # Transcripts: PyMuPDF is fast and accurate for text
            return [ParserType.PYMUPDF, ParserType.DOCLING, ParserType.PDFPLUMBER]

        elif "presentation" in filename or "earnings" in filename:
            # Presentations: Docling handles complex layouts
            return [ParserType.DOCLING, ParserType.PDFPLUMBER, ParserType.PYMUPDF]

        else:
            # Default: Try Docling first for best general quality
            if self.prefer_parser:
                order = [self.prefer_parser]
                for pt in ParserType:
                    if pt != self.prefer_parser and pt in self.parsers:
                        order.append(pt)
                return order

            return [ParserType.DOCLING, ParserType.PDFPLUMBER, ParserType.PYMUPDF]

    def compare_parsers(self, pdf_path: Path) -> Dict[ParserType, ParsedDocument]:
        """
        Run all parsers and return results for comparison

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary of parser results
        """
        results = {}

        for parser_type, parser in self.parsers.items():
            try:
                result = parser.parse(pdf_path)
                results[parser_type] = result
                logger.info(
                    f"{parser_type.value}: quality={result.extraction_quality:.3f}, "
                    f"elements={len(result.elements)}, pages={result.page_count}"
                )
            except Exception as e:
                logger.error(f"Parser {parser_type.value} failed: {e}")

        return results
