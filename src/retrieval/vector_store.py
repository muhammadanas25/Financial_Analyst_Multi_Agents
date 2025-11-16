"""
Vector store integration for Weaviate with hybrid search support.
Implements semantic + BM25 keyword search with α=0.3 for financial documents.
"""
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from config.config import settings


class WeaviateVectorStore:
    """
    Weaviate vector store with hybrid search capabilities.

    Features:
    - Hybrid search (semantic + BM25) with configurable alpha
    - Temporal filtering (fiscal_year, fiscal_quarter)
    - Rich metadata storage
    - Batch upsert for efficiency
    """

    COLLECTION_NAME = "FinancialDocument"

    def __init__(
        self,
        host: str = "http://localhost:8080",
        embedding_model: Optional[str] = None,
        use_openai_embeddings: bool = False
    ):
        """
        Initialize Weaviate vector store

        Args:
            host: Weaviate host URL
            embedding_model: SentenceTransformer model name
            use_openai_embeddings: Whether to use OpenAI embeddings instead of local
        """
        self.host = host
        self.use_openai_embeddings = use_openai_embeddings

        # Initialize Weaviate client
        self.client = weaviate.connect_to_local(host=host)

        # Initialize embedding model if using local embeddings
        if not use_openai_embeddings:
            model_name = embedding_model or settings.embedding_model
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.embedding_model = None
            logger.info("Using OpenAI embeddings")

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create Weaviate collection with schema"""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.COLLECTION_NAME):
                logger.info(f"Collection '{self.COLLECTION_NAME}' already exists")
                return

            # Define collection schema
            logger.info(f"Creating collection '{self.COLLECTION_NAME}'...")

            self.client.collections.create(
                name=self.COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="chunk_type", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),

                    # Document metadata
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="company_name", data_type=DataType.TEXT),
                    Property(name="ticker", data_type=DataType.TEXT),

                    # Temporal metadata
                    Property(name="fiscal_year", data_type=DataType.INT),
                    Property(name="fiscal_quarter", data_type=DataType.INT),
                    Property(name="report_type", data_type=DataType.TEXT),

                    # Financial metadata
                    Property(name="statement_type", data_type=DataType.TEXT),
                    Property(name="currency", data_type=DataType.TEXT),
                    Property(name="scale", data_type=DataType.TEXT),
                    Property(name="accounting_standard", data_type=DataType.TEXT),

                    # Quality indicators
                    Property(name="extraction_quality", data_type=DataType.NUMBER),
                    Property(name="has_numbers", data_type=DataType.BOOL),
                ]
            )

            logger.info(f"Collection '{self.COLLECTION_NAME}' created successfully")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.use_openai_embeddings:
            # TODO: Implement OpenAI embeddings
            raise NotImplementedError("OpenAI embeddings not yet implemented")
        else:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Add document chunks to vector store

        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for insertion
        """
        logger.info(f"Adding {len(chunks)} chunks to Weaviate...")

        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            with collection.batch.dynamic() as batch_insert:
                for chunk in batch:
                    # Generate embedding
                    embedding = self.embed_text(chunk["content"])

                    # Prepare metadata
                    metadata = chunk.get("metadata", {})

                    # Create object
                    properties = {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "chunk_type": chunk["chunk_type"],
                        "page_number": chunk["page_number"],

                        # Document metadata
                        "file_name": metadata.get("file_name", ""),
                        "company_name": metadata.get("company_name", ""),
                        "ticker": metadata.get("ticker", ""),

                        # Temporal metadata
                        "fiscal_year": metadata.get("fiscal_year", 0),
                        "fiscal_quarter": metadata.get("fiscal_quarter", 0),
                        "report_type": metadata.get("report_type", ""),

                        # Financial metadata
                        "statement_type": metadata.get("statement_type", ""),
                        "currency": metadata.get("currency", ""),
                        "scale": metadata.get("scale", ""),
                        "accounting_standard": metadata.get("accounting_standard", ""),

                        # Quality indicators
                        "extraction_quality": metadata.get("extraction_quality", 0.0),
                        "has_numbers": metadata.get("has_numbers", False),
                    }

                    batch_insert.add_object(
                        properties=properties,
                        vector=embedding
                    )

            logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        logger.info(f"Successfully added {len(chunks)} chunks to Weaviate")

    def hybrid_search(
        self,
        query: str,
        alpha: float = 0.3,
        limit: int = 10,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[int] = None,
        statement_type: Optional[str] = None,
        company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search.

        Formula: hybrid_score = α × vector_score + (1-α) × keyword_score
        For financial documents: α = 0.3 (keyword-heavy)

        Args:
            query: Search query
            alpha: Balance between semantic (1.0) and keyword (0.0) search
            limit: Maximum number of results
            fiscal_year: Filter by fiscal year
            fiscal_quarter: Filter by fiscal quarter
            statement_type: Filter by statement type
            company: Filter by company name

        Returns:
            List of search results with metadata
        """
        logger.info(f"Hybrid search: query='{query[:50]}...', alpha={alpha}, limit={limit}")

        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Generate query embedding
        query_vector = self.embed_text(query)

        # Build filters
        filters = []
        if fiscal_year is not None:
            filters.append(f"fiscal_year == {fiscal_year}")
        if fiscal_quarter is not None:
            filters.append(f"fiscal_quarter == {fiscal_quarter}")
        if statement_type:
            filters.append(f"statement_type == '{statement_type}'")
        if company:
            filters.append(f"company_name == '{company}'")

        # Combine filters with AND
        where_filter = None
        if filters:
            # Build Weaviate filter
            from weaviate.classes.query import Filter

            where_filter = Filter.by_property("fiscal_year").equal(fiscal_year) if fiscal_year else None

            if fiscal_quarter and where_filter:
                where_filter = where_filter & Filter.by_property("fiscal_quarter").equal(fiscal_quarter)
            elif fiscal_quarter:
                where_filter = Filter.by_property("fiscal_quarter").equal(fiscal_quarter)

        # Perform hybrid search
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            where=where_filter
        )

        # Format results
        results = []
        for obj in response.objects:
            result = {
                "chunk_id": obj.properties.get("chunk_id"),
                "content": obj.properties.get("content"),
                "chunk_type": obj.properties.get("chunk_type"),
                "page_number": obj.properties.get("page_number"),
                "score": obj.metadata.score,
                "metadata": {
                    "file_name": obj.properties.get("file_name"),
                    "company_name": obj.properties.get("company_name"),
                    "ticker": obj.properties.get("ticker"),
                    "fiscal_year": obj.properties.get("fiscal_year"),
                    "fiscal_quarter": obj.properties.get("fiscal_quarter"),
                    "report_type": obj.properties.get("report_type"),
                    "statement_type": obj.properties.get("statement_type"),
                    "currency": obj.properties.get("currency"),
                    "scale": obj.properties.get("scale"),
                    "extraction_quality": obj.properties.get("extraction_quality"),
                }
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")
        return results

    def delete_all(self):
        """Delete all objects from collection"""
        logger.warning("Deleting all objects from collection...")
        collection = self.client.collections.get(self.COLLECTION_NAME)
        collection.data.delete_many(where=None)
        logger.info("All objects deleted")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Get total count
        result = collection.aggregate.over_all(total_count=True)

        return {
            "total_objects": result.total_count,
            "collection_name": self.COLLECTION_NAME,
        }

    def close(self):
        """Close Weaviate connection"""
        self.client.close()
        logger.info("Weaviate connection closed")


def ingest_chunks_to_weaviate(
    chunks: List[Dict[str, Any]],
    host: str = "http://localhost:8080",
    embedding_model: Optional[str] = None
) -> WeaviateVectorStore:
    """
    Convenience function to ingest chunks into Weaviate

    Args:
        chunks: List of chunk dictionaries
        host: Weaviate host URL
        embedding_model: Embedding model name

    Returns:
        WeaviateVectorStore instance
    """
    store = WeaviateVectorStore(host=host, embedding_model=embedding_model)
    store.add_chunks(chunks)

    # Print stats
    stats = store.get_stats()
    logger.info(f"Weaviate stats: {stats}")

    return store
