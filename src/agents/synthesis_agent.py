"""
Synthesis Agent
Responsibilities:
- Combine outputs from specialist agents
- Generate coherent narrative
- Add inline citations
- Assign confidence scores
"""
from typing import Dict, Any, List
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import FinancialAnalysisState, log_agent_action
from config.config import settings


class SynthesisAgent:
    """
    Agent for synthesizing final response with citations.

    Combines information from:
    - Retrieved documents
    - Calculation results
    - Analysis results
    """

    def __init__(self):
        """Initialize synthesis agent"""
        self.llm = ChatOpenAI(
            model=settings.primary_llm_model,
            temperature=0.3  # Slight creativity for natural language
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst synthesizing a response to a user's query.

Your task:
1. Combine information from retrieved documents and calculations
2. Generate a clear, accurate response
3. Add inline citations [Document X, Page Y]
4. Maintain professional financial communication style
5. DO NOT make up information - only use provided context
6. If information is insufficient, clearly state limitations

Guidelines:
- Start with a direct answer to the question
- Support claims with specific numbers and citations
- Use clear, concise language
- For calculations, show the formula and result
- Indicate confidence level if uncertain
- Always cite sources for financial figures

Retrieved Context:
{context}

Calculations Performed:
{calculations}

User Query: {query}

Generate a comprehensive response with inline citations."""),
            ("user", "{query}")
        ])

    def __call__(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Synthesize final response

        Args:
            state: Current state

        Returns:
            Updated state with synthesized response
        """
        query = state["original_query"]
        retrieved_docs = state.get("retrieved_documents", [])
        calculations = state.get("calculations", {})

        logger.info(f"Synthesis Agent generating response for: {query}")

        # Prepare context
        context = self._format_context(retrieved_docs)
        calculations_text = self._format_calculations(calculations)

        try:
            # Generate response
            chain = self.prompt | self.llm
            response = chain.invoke({
                "query": query,
                "context": context,
                "calculations": calculations_text
            })

            synthesis = response.content

            # Extract citations
            citations = self._extract_citations(retrieved_docs)

            # Calculate confidence score
            confidence = self._calculate_confidence(state, retrieved_docs)

            # Update state
            state["synthesis"] = synthesis
            state["citations"] = citations
            state["confidence_score"] = confidence

            # Determine if human review needed
            if confidence < settings.confidence_threshold:
                state["needs_human_review"] = True
                logger.warning(f"Low confidence ({confidence:.2f}) - flagging for human review")

            # Add reasoning
            state["reasoning_steps"].append(
                f"Synthesized response with {len(citations)} citations (confidence: {confidence:.2f})"
            )

            # Log action
            state = log_agent_action(
                state,
                agent_name="SynthesisAgent",
                action="synthesize_response",
                details={
                    "num_citations": len(citations),
                    "confidence_score": confidence,
                    "needs_review": state["needs_human_review"],
                }
            )

            logger.info(f"Response synthesized (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state["error_message"] = f"Synthesis error: {e}"
            state["synthesis"] = f"Error generating response: {e}"

        return state

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for prompt"""
        if not retrieved_docs:
            return "No documents retrieved."

        formatted = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            formatted.append(
                f"[Document {i}]\n"
                f"Source: {metadata.get('file_name', 'Unknown')}, Page {doc.get('page_number', '?')}\n"
                f"Period: Q{metadata.get('fiscal_quarter', '?')} {metadata.get('fiscal_year', '?')}\n"
                f"Content: {doc.get('content', '')}\n"
                f"Relevance Score: {doc.get('score', 0):.3f}\n"
            )

        return "\n\n".join(formatted)

    def _format_calculations(self, calculations: Dict[str, Any]) -> str:
        """Format calculation results for prompt"""
        if not calculations:
            return "No calculations performed."

        calc_text = []
        for key, value in calculations.items():
            if isinstance(value, dict):
                calc_text.append(f"{key}: {value}")
            else:
                calc_text.append(f"{key}: {value}")

        return "\n".join(calc_text)

    def _extract_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved documents"""
        citations = []

        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            citation = {
                "source": metadata.get("file_name", "Unknown"),
                "page": doc.get("page_number"),
                "fiscal_year": metadata.get("fiscal_year"),
                "fiscal_quarter": metadata.get("fiscal_quarter"),
                "report_type": metadata.get("report_type"),
                "content_preview": doc.get("content", "")[:200] + "...",
                "score": doc.get("score", 0),
            }
            citations.append(citation)

        return citations

    def _calculate_confidence(
        self,
        state: FinancialAnalysisState,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence score

        Factors:
        - Retrieval score
        - Number of documents
        - Calculation verification
        - Temporal match
        """
        confidence_factors = []

        # Factor 1: Retrieval score
        retrieval_score = state.get("retrieval_score", 0.0)
        confidence_factors.append(min(retrieval_score * 2, 1.0))  # Normalize

        # Factor 2: Number of documents (more = higher confidence)
        num_docs = len(retrieved_docs)
        doc_confidence = min(num_docs / 5, 1.0)  # 5+ docs = max confidence
        confidence_factors.append(doc_confidence)

        # Factor 3: Calculation verification
        if state.get("calculation_verified", False):
            confidence_factors.append(1.0)
        elif state.get("calculation_errors"):
            confidence_factors.append(0.5)

        # Factor 4: Temporal match
        temporal_context = state.get("temporal_context", {})
        if temporal_context and retrieved_docs:
            # Check if retrieved docs match temporal filters
            matches = sum(
                1 for doc in retrieved_docs
                if doc.get("metadata", {}).get("fiscal_year") == temporal_context.get("fiscal_year")
            )
            temporal_confidence = matches / len(retrieved_docs) if retrieved_docs else 0
            confidence_factors.append(temporal_confidence)

        # Average all factors
        if confidence_factors:
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            overall_confidence = 0.5

        return round(overall_confidence, 3)


def create_synthesis_agent() -> SynthesisAgent:
    """
    Factory function to create synthesis agent

    Returns:
        SynthesisAgent instance
    """
    return SynthesisAgent()
