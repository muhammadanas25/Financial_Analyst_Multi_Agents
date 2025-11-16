"""
Complete document ingestion pipeline.
Orchestrates parsing, metadata extraction, and chunking.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger
import json

from .parsers import MultiParser, ParserType, ParsedDocument
from .metadata_extractor import extract_financial_metadata
from .chunker import chunk_financial_document, DocumentChunk


class DocumentIngestionPipeline:
    """
    Complete pipeline for ingesting financial documents.

    Workflow:
    1. Parse PDF with multi-parser strategy
    2. Extract metadata
    3. Chunk document with element-based strategy
    4. Prepare for embedding and indexing
    """

    def __init__(
        self,
        output_dir: Path = Path("./output"),
        prefer_parser: Optional[ParserType] = None,
        max_chunk_size: int = 2048,
        chunk_overlap: int = 200,
    ):
        """
        Initialize ingestion pipeline

        Args:
            output_dir: Directory for output files
            prefer_parser: Preferred parser to try first
            max_chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.parser = MultiParser(prefer_parser=prefer_parser)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_document(
        self,
        pdf_path: Path,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF document

        Args:
            pdf_path: Path to PDF file
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"=== Starting ingestion of {pdf_path.name} ===")

        # Step 1: Parse document
        logger.info("Step 1: Parsing document...")
        parsed_doc = self.parser.parse(pdf_path)

        logger.info(
            f"Parsed {parsed_doc.page_count} pages, "
            f"{len(parsed_doc.elements)} elements, "
            f"quality: {parsed_doc.extraction_quality:.3f}"
        )

        # Step 2: Extract metadata
        logger.info("Step 2: Extracting metadata...")
        metadata = extract_financial_metadata(parsed_doc, pdf_path.name)

        logger.info(
            f"Extracted metadata: {metadata['company_name']} - "
            f"{metadata['report_type']} - "
            f"Q{metadata.get('fiscal_quarter', 'N/A')} {metadata.get('fiscal_year', 'N/A')}"
        )

        # Step 3: Chunk document
        logger.info("Step 3: Chunking document...")
        chunks = chunk_financial_document(
            parsed_doc,
            pdf_path.name,
            chunker_type="element-based",
            max_chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        logger.info(
            f"Created {len(chunks)} chunks "
            f"({metadata.get('chunk_statistics', {}).get('table_chunks', 0)} tables, "
            f"{metadata.get('chunk_statistics', {}).get('text_chunks', 0)} text)"
        )

        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(pdf_path.stem, parsed_doc, metadata, chunks)

        result = {
            "document_name": pdf_path.name,
            "parsed_document": parsed_doc,
            "metadata": metadata,
            "chunks": chunks,
            "summary": {
                "page_count": parsed_doc.page_count,
                "element_count": len(parsed_doc.elements),
                "chunk_count": len(chunks),
                "extraction_quality": parsed_doc.extraction_quality,
                "parser_used": parsed_doc.parser_used.value,
                "company": metadata.get("company_name"),
                "fiscal_year": metadata.get("fiscal_year"),
                "fiscal_quarter": metadata.get("fiscal_quarter"),
                "report_type": metadata.get("report_type"),
            }
        }

        logger.info(f"=== Completed ingestion of {pdf_path.name} ===\n")

        return result

    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "*.pdf",
        save_intermediate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ingest all PDFs in a directory

        Args:
            directory: Directory containing PDFs
            pattern: File pattern to match
            save_intermediate: Whether to save intermediate results

        Returns:
            List of ingestion results
        """
        pdf_files = list(Path(directory).glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        results = []
        for pdf_path in pdf_files:
            try:
                result = self.ingest_document(pdf_path, save_intermediate=save_intermediate)
                results.append(result)
            except Exception as e:
                logger.error(f"Error ingesting {pdf_path.name}: {e}")
                continue

        # Save summary
        self._save_ingestion_summary(results)

        return results

    def compare_parsers(self, pdf_path: Path) -> Dict[str, ParsedDocument]:
        """
        Compare all parsers on a single document

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary of parser results
        """
        logger.info(f"=== Comparing parsers on {pdf_path.name} ===")

        results = self.parser.compare_parsers(pdf_path)

        # Save comparison
        self._save_parser_comparison(pdf_path.stem, results)

        return results

    def _save_intermediate_results(
        self,
        doc_name: str,
        parsed_doc: ParsedDocument,
        metadata: Dict,
        chunks: List[DocumentChunk]
    ):
        """Save intermediate processing results"""
        doc_output_dir = self.output_dir / doc_name
        doc_output_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata_path = doc_output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save chunks
        chunks_path = doc_output_dir / "chunks.json"
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,  # Truncate for readability
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "metadata": chunk.metadata,
            }
            chunks_data.append(chunk_dict)

        with open(chunks_path, 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)

        # Save full chunks (for embedding)
        chunks_full_path = doc_output_dir / "chunks_full.json"
        chunks_full_data = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "chunk_type": c.chunk_type,
                "page_number": c.page_number,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        with open(chunks_full_path, 'w') as f:
            json.dump(chunks_full_data, f, indent=2, default=str)

        # Save extraction quality report
        report_path = doc_output_dir / "extraction_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Extraction Report for {doc_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Parser Used: {parsed_doc.parser_used.value}\n")
            f.write(f"Quality Score: {parsed_doc.extraction_quality:.3f}\n")
            f.write(f"Page Count: {parsed_doc.page_count}\n")
            f.write(f"Element Count: {len(parsed_doc.elements)}\n")
            f.write(f"Chunk Count: {len(chunks)}\n\n")
            f.write(f"Metadata:\n")
            f.write(f"  Company: {metadata.get('company_name', 'N/A')}\n")
            f.write(f"  Report Type: {metadata.get('report_type', 'N/A')}\n")
            f.write(f"  Fiscal Year: {metadata.get('fiscal_year', 'N/A')}\n")
            f.write(f"  Fiscal Quarter: Q{metadata.get('fiscal_quarter', 'N/A')}\n")
            f.write(f"  Currency: {metadata.get('currency', 'N/A')}\n")
            f.write(f"  Scale: {metadata.get('scale', 'N/A')}\n")

        logger.info(f"Saved intermediate results to {doc_output_dir}")

    def _save_parser_comparison(self, doc_name: str, results: Dict[ParserType, ParsedDocument]):
        """Save parser comparison results"""
        comparison_path = self.output_dir / f"{doc_name}_parser_comparison.json"

        comparison_data = {}
        for parser_type, parsed_doc in results.items():
            comparison_data[parser_type.value] = {
                "quality": parsed_doc.extraction_quality,
                "page_count": parsed_doc.page_count,
                "element_count": len(parsed_doc.elements),
                "table_count": sum(1 for e in parsed_doc.elements if e.type.value == "table"),
            }

        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        logger.info(f"Saved parser comparison to {comparison_path}")

    def _save_ingestion_summary(self, results: List[Dict[str, Any]]):
        """Save summary of all ingested documents"""
        summary_path = self.output_dir / "ingestion_summary.json"

        summary = {
            "total_documents": len(results),
            "documents": [r["summary"] for r in results],
            "statistics": {
                "total_pages": sum(r["summary"]["page_count"] for r in results),
                "total_chunks": sum(r["summary"]["chunk_count"] for r in results),
                "avg_quality": sum(r["summary"]["extraction_quality"] for r in results) / len(results) if results else 0,
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved ingestion summary to {summary_path}")


# Convenience function
def ingest_financial_documents(
    pdf_paths: List[Path],
    output_dir: Path = Path("./output"),
    prefer_parser: Optional[ParserType] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest multiple documents

    Args:
        pdf_paths: List of PDF paths
        output_dir: Output directory
        prefer_parser: Preferred parser

    Returns:
        List of ingestion results
    """
    pipeline = DocumentIngestionPipeline(
        output_dir=output_dir,
        prefer_parser=prefer_parser,
    )

    results = []
    for pdf_path in pdf_paths:
        result = pipeline.ingest_document(pdf_path)
        results.append(result)

    return results
