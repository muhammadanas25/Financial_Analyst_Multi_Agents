"""
Metadata extraction for financial documents.
Extracts company info, temporal data, accounting standards, etc.
"""
import re
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
from loguru import logger
import dateparser
from .parsers import ParsedDocument, ElementType


class FinancialMetadataExtractor:
    """Extract structured metadata from financial documents"""

    # Common fiscal quarter patterns
    QUARTER_PATTERNS = [
        r'Q([1-4])\s*(\d{4})',  # Q1 2024
        r'([1-4])Q\s*(\d{4})',  # 1Q 2024
        r'([1-4])\s*quarter\s*(\d{4})',  # 1st quarter 2024
        r'quarter\s*([1-4])\s*(\d{4})',
    ]

    # Fiscal year patterns
    YEAR_PATTERNS = [
        r'FY\s*(\d{4})',  # FY 2024
        r'fiscal\s*year\s*(\d{4})',
        r'year\s*ended?\s*.*?(\d{4})',
    ]

    # Currency patterns
    CURRENCY_PATTERNS = {
        'AED': r'\bAED\b|UAE\s*Dirham',
        'USD': r'\$|USD|US\s*Dollar',
        'EUR': r'€|EUR|Euro',
        'GBP': r'£|GBP|Pound',
    }

    # Scale patterns
    SCALE_PATTERNS = {
        'thousands': r'\(\s*in\s*thousands\s*\)|\(000\'?s?\)',
        'millions': r'\(\s*in\s*millions\s*\)|\(AED\s*m\)|\(USD\s*m\)',
        'billions': r'\(\s*in\s*billions\s*\)|\(AED\s*bn\)|\(USD\s*bn\)',
    }

    def __init__(self):
        self.company_keywords = {
            'FAB': ['First Abu Dhabi Bank', 'FAB', 'NBAD', 'FGB'],
            # Add more companies as needed
        }

    def extract_metadata(self, parsed_doc: ParsedDocument, filename: str) -> Dict:
        """
        Extract comprehensive metadata from parsed document

        Args:
            parsed_doc: Parsed document with elements
            filename: Original filename

        Returns:
            Dictionary of metadata
        """
        logger.info(f"Extracting metadata from {filename}...")

        # Get first few pages of text for metadata extraction
        text_content = self._get_header_text(parsed_doc, max_pages=3)

        metadata = {
            # Basic file info
            "file_name": filename,
            "file_path": parsed_doc.metadata.get("file_path", ""),
            "parser_used": parsed_doc.parser_used.value,
            "page_count": parsed_doc.page_count,
            "extraction_quality": parsed_doc.extraction_quality,

            # Company identification
            "company_name": self._extract_company_name(text_content, filename),
            "ticker": self._extract_ticker(text_content, filename),

            # Temporal information
            "fiscal_year": self._extract_fiscal_year(text_content, filename),
            "fiscal_quarter": self._extract_fiscal_quarter(text_content, filename),
            "reporting_period_start": self._extract_period_start(text_content),
            "reporting_period_end": self._extract_period_end(text_content),
            "filing_date": self._extract_filing_date(text_content, filename),

            # Document classification
            "report_type": self._classify_report_type(text_content, filename),
            "document_sections": self._identify_sections(parsed_doc),

            # Accounting standards
            "accounting_standard": self._detect_accounting_standard(text_content),
            "currency": self._extract_currency(text_content),
            "scale": self._extract_scale(text_content),

            # Quality indicators
            "has_tables": self._has_tables(parsed_doc),
            "table_count": self._count_tables(parsed_doc),
            "element_count": len(parsed_doc.elements),
        }

        logger.info(f"Extracted metadata: {metadata['company_name']} - {metadata['report_type']} - Q{metadata.get('fiscal_quarter', 'N/A')} {metadata.get('fiscal_year', 'N/A')}")

        return metadata

    def _get_header_text(self, parsed_doc: ParsedDocument, max_pages: int = 3) -> str:
        """Get text from first few pages for metadata extraction"""
        text_parts = []

        for element in parsed_doc.elements:
            if element.page <= max_pages:
                text_parts.append(element.content)

        return "\n".join(text_parts)

    def _extract_company_name(self, text: str, filename: str) -> Optional[str]:
        """Extract company name"""
        # Check filename first
        if 'FAB' in filename.upper():
            return "First Abu Dhabi Bank"

        # Search in text
        for company, patterns in self.company_keywords.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if company == 'FAB':
                        return "First Abu Dhabi Bank"

        # Try to find company name in first few lines
        lines = text.split('\n')[:10]
        for line in lines:
            # Look for "Bank" or "Company" in capitalized text
            if re.search(r'[A-Z][a-z]+\s+(Bank|Company|Corporation|Group)', line):
                match = re.search(r'([A-Z][A-Za-z\s]+(?:Bank|Company|Corporation|Group))', line)
                if match:
                    return match.group(1).strip()

        return None

    def _extract_ticker(self, text: str, filename: str) -> Optional[str]:
        """Extract stock ticker symbol"""
        # FAB-specific
        if 'FAB' in filename.upper() or 'FAB' in text[:1000]:
            return "FAB"

        # Generic ticker pattern (3-5 uppercase letters)
        match = re.search(r'\b([A-Z]{3,5})\b', text[:1000])
        if match:
            return match.group(1)

        return None

    def _extract_fiscal_year(self, text: str, filename: str) -> Optional[int]:
        """Extract fiscal year"""
        # Try filename first
        filename_match = re.search(r'(\d{4})', filename)
        if filename_match:
            year = int(filename_match.group(1))
            if 2000 <= year <= 2030:
                return year

        # Try patterns in text
        for pattern in self.YEAR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2000 <= year <= 2030:
                    return year

        # Look for any 4-digit year
        years = re.findall(r'\b(20\d{2})\b', text)
        if years:
            return int(years[0])

        return None

    def _extract_fiscal_quarter(self, text: str, filename: str) -> Optional[int]:
        """Extract fiscal quarter (1-4)"""
        # Try filename first
        for pattern in self.QUARTER_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Try text
        for pattern in self.QUARTER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _extract_period_start(self, text: str) -> Optional[str]:
        """Extract reporting period start date"""
        patterns = [
            r'period\s+from\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'three\s+months\s+ended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')

        return None

    def _extract_period_end(self, text: str) -> Optional[str]:
        """Extract reporting period end date"""
        patterns = [
            r'period\s+ended?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'as\s+of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'for\s+the\s+(?:three|six|nine|twelve)\s+months\s+ended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')

        return None

    def _extract_filing_date(self, text: str, filename: str) -> Optional[str]:
        """Extract filing/publication date"""
        # Try to find date in text
        patterns = [
            r'filed?\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'published\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')

        return None

    def _classify_report_type(self, text: str, filename: str) -> str:
        """Classify document type"""
        filename_lower = filename.lower()

        # Check filename patterns
        if 'earnings' in filename_lower and 'presentation' in filename_lower:
            return "earnings_presentation"
        elif 'fs' in filename_lower or 'financial-statement' in filename_lower:
            return "financial_statements"
        elif 'call' in filename_lower or 'transcript' in filename_lower:
            return "earnings_call_transcript"
        elif '10-k' in filename_lower:
            return "10-K"
        elif '10-q' in filename_lower:
            return "10-Q"
        elif '8-k' in filename_lower:
            return "8-K"
        elif 'annual' in filename_lower:
            return "annual_report"
        elif 'quarterly' in filename_lower or 'quarter' in filename_lower:
            return "quarterly_report"

        # Check text content
        text_lower = text.lower()
        if 'consolidated financial statements' in text_lower:
            return "financial_statements"
        elif 'earnings call' in text_lower:
            return "earnings_call_transcript"

        return "unknown"

    def _identify_sections(self, parsed_doc: ParsedDocument) -> List[str]:
        """Identify major sections in document"""
        sections = []

        for element in parsed_doc.elements:
            if element.type == ElementType.TITLE:
                # Clean and add section title
                section = element.content.strip()
                if section and len(section) < 200:
                    sections.append(section)

        # Return unique sections (first 50)
        return list(dict.fromkeys(sections))[:50]

    def _detect_accounting_standard(self, text: str) -> Optional[str]:
        """Detect accounting standard (GAAP, IFRS, etc.)"""
        if re.search(r'\bIFRS\b', text, re.IGNORECASE):
            return "IFRS"
        elif re.search(r'\bGAAP\b', text, re.IGNORECASE):
            return "GAAP"
        elif re.search(r'\bUS\s*GAAP\b', text, re.IGNORECASE):
            return "US-GAAP"

        return None

    def _extract_currency(self, text: str) -> Optional[str]:
        """Extract primary currency"""
        for currency, pattern in self.CURRENCY_PATTERNS.items():
            if re.search(pattern, text[:2000], re.IGNORECASE):
                return currency

        return None

    def _extract_scale(self, text: str) -> Optional[str]:
        """Extract numerical scale (thousands, millions, billions)"""
        for scale, pattern in self.SCALE_PATTERNS.items():
            if re.search(pattern, text[:2000], re.IGNORECASE):
                return scale

        return "units"  # Default to units if not specified

    def _has_tables(self, parsed_doc: ParsedDocument) -> bool:
        """Check if document has tables"""
        return any(e.type == ElementType.TABLE for e in parsed_doc.elements)

    def _count_tables(self, parsed_doc: ParsedDocument) -> int:
        """Count tables in document"""
        return sum(1 for e in parsed_doc.elements if e.type == ElementType.TABLE)


def extract_financial_metadata(parsed_doc: ParsedDocument, filename: str) -> Dict:
    """
    Convenience function to extract metadata

    Args:
        parsed_doc: Parsed document
        filename: Original filename

    Returns:
        Dictionary of metadata
    """
    extractor = FinancialMetadataExtractor()
    return extractor.extract_metadata(parsed_doc, filename)
