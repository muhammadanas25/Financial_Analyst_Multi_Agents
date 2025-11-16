"""
Script to ingest FAB documents into Weaviate vector database.

This script:
1. Processes FAB Q1 2025 PDFs
2. Generates embeddings
3. Stores in Weaviate for retrieval
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.document_processing.ingestion_pipeline import DocumentIngestionPipeline
from src.retrieval.vector_store import WeaviateVectorStore, ingest_chunks_to_weaviate
from config.config import settings

# Configure logger
logger.add(
    "logs/ingestion_{time}.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def main():
    """Ingest FAB documents to Weaviate"""

    logger.info("=" * 80)
    logger.info("FAB Document Ingestion to Weaviate")
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

    # Step 1: Process documents (parse, extract metadata, chunk)
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Processing Documents")
    logger.info("=" * 80)

    pipeline = DocumentIngestionPipeline(
        output_dir=output_dir,
        max_chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []
    for doc_path in fab_documents:
        logger.info(f"\nProcessing: {doc_path.name}")

        try:
            result = pipeline.ingest_document(doc_path, save_intermediate=True)

            # Convert chunks to dictionaries
            chunks = result["chunks"]
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                }
                chunk_dicts.append(chunk_dict)

            all_chunks.extend(chunk_dicts)

            logger.info(f"✓ Processed {doc_path.name}: {len(chunk_dicts)} chunks")

        except Exception as e:
            logger.error(f"Error processing {doc_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    logger.info(f"\nTotal chunks to ingest: {len(all_chunks)}")

    # Step 2: Initialize Weaviate
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Connecting to Weaviate")
    logger.info("=" * 80)

    try:
        vector_store = WeaviateVectorStore(
            host="http://localhost:8080",
            embedding_model=settings.embedding_model,
            use_openai_embeddings=False
        )

        logger.info("✓ Connected to Weaviate")

        # Check existing data
        stats = vector_store.get_stats()
        logger.info(f"Existing objects in database: {stats['total_objects']}")

        # Optional: Clear existing data
        if stats['total_objects'] > 0:
            response = input("\nDatabase contains existing data. Clear it? (yes/no): ")
            if response.lower() == 'yes':
                vector_store.delete_all()
                logger.info("✓ Database cleared")

    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        logger.error("\nMake sure Weaviate is running:")
        logger.error("  docker-compose up -d")
        return

    # Step 3: Ingest chunks
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Ingesting Chunks to Weaviate")
    logger.info("=" * 80)

    try:
        vector_store.add_chunks(all_chunks, batch_size=100)

        # Get final stats
        stats = vector_store.get_stats()

        logger.info("\n" + "=" * 80)
        logger.info("Ingestion Complete!")
        logger.info("=" * 80)
        logger.info(f"Total chunks ingested: {len(all_chunks)}")
        logger.info(f"Total objects in database: {stats['total_objects']}")

        # Test retrieval
        logger.info("\n" + "=" * 80)
        logger.info("Testing Retrieval")
        logger.info("=" * 80)

        test_query = "What was FAB's total revenue in Q1 2025?"
        logger.info(f"Test query: {test_query}")

        results = vector_store.hybrid_search(
            query=test_query,
            alpha=0.3,
            limit=3,
            fiscal_year=2025,
            fiscal_quarter=1
        )

        logger.info(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(
                f"\n{i}. Score: {result['score']:.3f} | "
                f"{result['metadata']['file_name']} | "
                f"Page {result['page_number']}"
            )
            logger.info(f"   Content preview: {result['content'][:150]}...")

        logger.info("\n" + "=" * 80)
        logger.info("✓ System ready for queries!")
        logger.info("=" * 80)

        # Close connection
        vector_store.close()

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
