"""
Example script to process FAB Q1 2025 financial documents.
Demonstrates the complete ingestion pipeline with multiple parsers.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.document_processing.ingestion_pipeline import DocumentIngestionPipeline
from src.document_processing.parsers import ParserType
from src.tools.financial_calculators import calculator

# Configure logger
logger.add(
    "logs/fab_processing_{time}.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def main():
    """Process FAB Q1 2025 documents"""

    logger.info("=" * 80)
    logger.info("FAB Q1 2025 Document Processing")
    logger.info("=" * 80)

    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    # FAB documents
    fab_documents = [
        data_dir / "FAB-Earnings-Presentation-Q1-2025.pdf",
        data_dir / "FAB-FS-Q1-2025-English.pdf",
        data_dir / "FAB-Q1-2025-Results-Call.pdf",
    ]

    # Check if documents exist
    for doc_path in fab_documents:
        if not doc_path.exists():
            logger.error(f"Document not found: {doc_path}")
            return

    # Initialize pipeline
    pipeline = DocumentIngestionPipeline(
        output_dir=output_dir,
        prefer_parser=None,  # Let pipeline choose best parser
        max_chunk_size=2048,
        chunk_overlap=200,
    )

    # Process each document
    results = []
    for doc_path in fab_documents:
        logger.info(f"\nProcessing: {doc_path.name}")
        logger.info("-" * 80)

        try:
            result = pipeline.ingest_document(doc_path, save_intermediate=True)
            results.append(result)

            # Print summary
            summary = result["summary"]
            logger.info(f"\nSummary for {doc_path.name}:")
            logger.info(f"  Company: {summary['company']}")
            logger.info(f"  Report Type: {summary['report_type']}")
            logger.info(f"  Fiscal Period: Q{summary['fiscal_quarter']} {summary['fiscal_year']}")
            logger.info(f"  Parser Used: {summary['parser_used']}")
            logger.info(f"  Quality Score: {summary['extraction_quality']:.3f}")
            logger.info(f"  Pages: {summary['page_count']}")
            logger.info(f"  Elements: {summary['element_count']}")
            logger.info(f"  Chunks: {summary['chunk_count']}")

        except Exception as e:
            logger.error(f"Error processing {doc_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Print overall statistics
    logger.info("\n" + "=" * 80)
    logger.info("Overall Statistics")
    logger.info("=" * 80)

    if results:
        total_pages = sum(r["summary"]["page_count"] for r in results)
        total_chunks = sum(r["summary"]["chunk_count"] for r in results)
        avg_quality = sum(r["summary"]["extraction_quality"] for r in results) / len(results)

        logger.info(f"Documents Processed: {len(results)}")
        logger.info(f"Total Pages: {total_pages}")
        logger.info(f"Total Chunks: {total_chunks}")
        logger.info(f"Average Quality: {avg_quality:.3f}")

        logger.info(f"\nOutput saved to: {output_dir}")

    # Example: Test financial calculator
    logger.info("\n" + "=" * 80)
    logger.info("Testing Financial Calculator")
    logger.info("=" * 80)

    # Example percentage change calculation
    logger.info("\nExample: Calculate Q1 2025 vs Q1 2024 revenue growth")
    result = calculator.calculate_percentage_change(
        current_value=5200,  # Example: 5.2 billion
        prior_value=4800,    # Example: 4.8 billion
        label="Q1 2025 vs Q1 2024 revenue growth"
    )

    logger.info(f"Result: {result['formatted']}")
    logger.info(f"Absolute Change: {result['absolute_change']:,.0f}")
    logger.info(f"Direction: {result['direction']}")
    logger.info(f"Verified: {result['verified']}")

    # Example number extraction
    logger.info("\nExample: Extract number from text")
    result = calculator.extract_number_from_text("AED 5.2 billion")
    logger.info(f"Extracted Value: {result['formatted']}")
    logger.info(f"Currency: {result['currency']}")
    logger.info(f"Scale: {result['scale']}")

    logger.info("\n" + "=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)


def compare_parsers_example():
    """Example: Compare parser performance"""

    logger.info("=" * 80)
    logger.info("Parser Comparison Example")
    logger.info("=" * 80)

    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    # Choose a document to compare parsers
    doc_path = data_dir / "FAB-FS-Q1-2025-English.pdf"

    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return

    pipeline = DocumentIngestionPipeline(output_dir=output_dir)

    logger.info(f"Comparing parsers on: {doc_path.name}\n")

    results = pipeline.compare_parsers(doc_path)

    # Print comparison
    logger.info("Parser Comparison Results:")
    logger.info("-" * 80)

    for parser_type, parsed_doc in results.items():
        logger.info(f"\n{parser_type.value.upper()}:")
        logger.info(f"  Quality: {parsed_doc.extraction_quality:.3f}")
        logger.info(f"  Pages: {parsed_doc.page_count}")
        logger.info(f"  Elements: {len(parsed_doc.elements)}")
        table_count = sum(1 for e in parsed_doc.elements if e.type.value == "table")
        logger.info(f"  Tables: {table_count}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # Run main processing
    main()

    # Optionally run parser comparison
    # Uncomment the line below to compare parsers
    # compare_parsers_example()
