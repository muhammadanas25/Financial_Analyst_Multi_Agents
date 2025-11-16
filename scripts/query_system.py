"""
Query the FAB financial analysis system.

This script demonstrates end-to-end querying with the multi-agent system.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.retrieval.vector_store import WeaviateVectorStore
from src.agents.workflow import create_workflow
from config.config import settings

# Configure logger
logger.add(
    "logs/query_{time}.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


# Example queries covering different types
EXAMPLE_QUERIES = [
    # Simple retrieval
    "What was FAB's total revenue in Q1 2025?",

    # Calculation
    "Calculate FAB's revenue growth from Q1 2024 to Q1 2025",

    # Comparison
    "Compare FAB's net income in Q1 2025 vs Q1 2024",

    # Analysis
    "What were the key drivers of FAB's performance in Q1 2025?",
]


def main():
    """Run interactive query system"""

    logger.info("=" * 80)
    logger.info("FAB Financial Analysis Query System")
    logger.info("=" * 80)

    # Connect to Weaviate
    logger.info("\nConnecting to Weaviate...")
    try:
        vector_store = WeaviateVectorStore(
            host="http://localhost:8080",
            embedding_model=settings.embedding_model,
            use_openai_embeddings=False
        )

        stats = vector_store.get_stats()
        logger.info(f"✓ Connected to Weaviate ({stats['total_objects']} documents)")

        if stats['total_objects'] == 0:
            logger.error("\n❌ No documents in database!")
            logger.error("Please run ingestion first:")
            logger.error("  python scripts/ingest_to_weaviate.py")
            return

    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        logger.error("\nMake sure Weaviate is running:")
        logger.error("  docker-compose up -d")
        return

    # Create workflow
    logger.info("Initializing multi-agent workflow...")
    workflow = create_workflow(vector_store)
    logger.info("✓ Workflow ready\n")

    # Interactive mode or example queries
    print("=" * 80)
    print("FAB Financial Analysis System")
    print("=" * 80)
    print("\nOptions:")
    print("1. Run example queries")
    print("2. Interactive mode")
    print("3. Exit")

    choice = input("\nChoice (1-3): ").strip()

    if choice == "1":
        run_example_queries(workflow)
    elif choice == "2":
        interactive_mode(workflow)
    else:
        print("Exiting...")

    # Close connection
    vector_store.close()


def run_example_queries(workflow):
    """Run predefined example queries"""

    print("\n" + "=" * 80)
    print("Running Example Queries")
    print("=" * 80)

    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}/{len(EXAMPLE_QUERIES)}")
        print(f"{'=' * 80}")
        print(f"Q: {query}\n")

        try:
            # Process query
            state = workflow.query(query)

            # Get formatted response
            response = workflow.get_response(state)

            print(f"A: {response}")

            # Show reasoning steps
            print(f"\nReasoning Steps:")
            for step in state.get("reasoning_steps", []):
                print(f"  - {step}")

            # Show agent sequence
            print(f"\nAgent Sequence: {' → '.join(state.get('agent_sequence', []))}")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"❌ Error: {e}")

        # Pause between queries
        if i < len(EXAMPLE_QUERIES):
            input("\nPress Enter for next query...")


def interactive_mode(workflow):
    """Interactive query mode"""

    print("\n" + "=" * 80)
    print("Interactive Query Mode")
    print("=" * 80)
    print("\nEnter your questions about FAB's Q1 2025 financials.")
    print("Type 'exit' to quit, 'examples' to see example queries.\n")

    while True:
        query = input("\nYour question: ").strip()

        if not query:
            continue

        if query.lower() == 'exit':
            print("Exiting...")
            break

        if query.lower() == 'examples':
            print("\nExample queries:")
            for i, ex in enumerate(EXAMPLE_QUERIES, 1):
                print(f"{i}. {ex}")
            continue

        print(f"\n{'=' * 80}")
        print("Processing...")
        print(f"{'=' * 80}\n")

        try:
            # Process query
            state = workflow.query(query)

            # Get formatted response
            response = workflow.get_response(state)

            print(f"\n{response}")

            # Optional: Show details
            show_details = input("\nShow details? (y/n): ").strip().lower()
            if show_details == 'y':
                print(f"\nIntent: {state.get('intent')}")
                print(f"Complexity: {state.get('query_complexity')}")
                print(f"Documents Retrieved: {len(state.get('retrieved_documents', []))}")
                print(f"Confidence: {state.get('confidence_score', 0):.1%}")

                print(f"\nReasoning Steps:")
                for step in state.get("reasoning_steps", []):
                    print(f"  - {step}")

                print(f"\nAgent Sequence: {' → '.join(state.get('agent_sequence', []))}")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Check if .env exists
    from pathlib import Path
    env_file = Path(__file__).parent.parent / ".env"

    if not env_file.exists():
        print("❌ Error: .env file not found!")
        print("\nPlease create .env file:")
        print("  cp .env.template .env")
        print("  # Edit .env and add your OPENAI_API_KEY")
        sys.exit(1)

    main()
