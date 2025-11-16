"""
Retrieval Agent
Responsibilities:
- Hybrid search (semantic + keyword)
- Temporal filtering
- Document selection and ranking
- Context assembly
"""
from typing import Dict, Any, List
from loguru import logger

from .state import FinancialAnalysisState, log_agent_action
from src.retrieval.vector_store import WeaviateVectorStore
from config.config import settings


class RetrievalAgent:
    """Agent for retrieving relevant documents using hybrid search"""

    def __init__(self, vector_store: WeaviateVectorStore):
        """
        Initialize retrieval agent

        Args:
            vector_store: Weaviate vector store instance
        """
        self.vector_store = vector_store
        self.hybrid_alpha = settings.hybrid_search_alpha  # Default 0.3 for financial docs
        self.top_k = settings.retrieval_top_k  # Default 10

    def __call__(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Retrieve relevant documents

        Args:
            state: Current state

        Returns:
            Updated state with retrieved documents
        """
        query = state["original_query"]
        temporal_context = state.get("temporal_context", {})
        intent = state.get("intent", "")

        logger.info(f"Retrieval Agent processing query: {query}")
        logger.info(f"Temporal context: {temporal_context}")

        # Perform hybrid search
        results = self._hybrid_search(query, temporal_context, intent)

        # Update state
        state["retrieved_documents"] = results

        # Extract metadata
        metadata = [r["metadata"] for r in results]
        state["document_metadata"] = metadata

        # Calculate average retrieval score
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            state["retrieval_score"] = avg_score
        else:
            state["retrieval_score"] = 0.0

        # Add reasoning step
        state["reasoning_steps"].append(
            f"Retrieved {len(results)} relevant documents with average score {state['retrieval_score']:.3f}"
        )

        # Log action
        state = log_agent_action(
            state,
            agent_name="RetrievalAgent",
            action="hybrid_search",
            details={
                "num_results": len(results),
                "avg_score": state["retrieval_score"],
                "temporal_filter": temporal_context,
                "alpha": self.hybrid_alpha,
            }
        )

        logger.info(f"Retrieved {len(results)} documents (avg score: {state['retrieval_score']:.3f})")

        return state

    def _hybrid_search(
        self,
        query: str,
        temporal_context: Dict[str, Any],
        intent: str
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with temporal filtering

        Args:
            query: Search query
            temporal_context: Temporal filters
            intent: Query intent for context

        Returns:
            List of search results
        """
        # Extract filters from temporal context
        fiscal_year = temporal_context.get("fiscal_year")
        fiscal_quarter = temporal_context.get("fiscal_quarter")

        # Detect statement type from query
        statement_type = self._detect_statement_type(query)

        # For comparisons, we need multiple searches
        if temporal_context.get("is_comparison") and "comparison_periods" in temporal_context:
            return self._comparison_retrieval(query, temporal_context, statement_type)

        # Single search
        results = self.vector_store.hybrid_search(
            query=query,
            alpha=self.hybrid_alpha,
            limit=self.top_k,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            statement_type=statement_type
        )

        return results

    def _comparison_retrieval(
        self,
        query: str,
        temporal_context: Dict[str, Any],
        statement_type: str
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for comparison queries"""
        periods = temporal_context["comparison_periods"]

        all_results = []

        # Search for each period
        for period in periods:
            results = self.vector_store.hybrid_search(
                query=query,
                alpha=self.hybrid_alpha,
                limit=self.top_k // 2,  # Split limit between periods
                fiscal_year=period.get("year"),
                fiscal_quarter=period.get("quarter"),
                statement_type=statement_type
            )

            # Tag results with period
            for r in results:
                r["comparison_period"] = period

            all_results.extend(results)

        return all_results

    def _detect_statement_type(self, query: str) -> str:
        """Detect financial statement type from query"""
        query_lower = query.lower()

        if any(term in query_lower for term in ['income statement', 'profit', 'loss', 'revenue', 'earnings']):
            return "income_statement"
        elif any(term in query_lower for term in ['balance sheet', 'assets', 'liabilities', 'equity']):
            return "balance_sheet"
        elif any(term in query_lower for term in ['cash flow', 'operating activities', 'investing', 'financing']):
            return "cash_flow_statement"

        return None  # No specific statement type


def create_retrieval_agent(vector_store: WeaviateVectorStore) -> RetrievalAgent:
    """
    Factory function to create retrieval agent

    Args:
        vector_store: Weaviate vector store

    Returns:
        RetrievalAgent instance
    """
    return RetrievalAgent(vector_store)
