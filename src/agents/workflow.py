"""
LangGraph workflow for multi-agent financial analysis.
Orchestrates all agents in a structured graph.
"""
from typing import Dict, Any, Literal
from loguru import logger
from langgraph.graph import StateGraph, END

from .state import FinancialAnalysisState, create_initial_state
from .input_validation_agent import InputValidationAgent
from .retrieval_agent import RetrievalAgent, create_retrieval_agent
from .calculation_agent import CalculationAgent, create_calculation_agent
from .synthesis_agent import SynthesisAgent, create_synthesis_agent
from src.retrieval.vector_store import WeaviateVectorStore


class FinancialAnalysisWorkflow:
    """
    Complete multi-agent workflow for financial document analysis.

    Agent Flow:
    1. Input Validation → Classify and extract temporal context
    2. Retrieval → Get relevant documents with hybrid search
    3. Calculation (if needed) → Perform numerical operations
    4. Synthesis → Generate final response with citations
    5. QA/Compliance → Final validation (optional)
    """

    def __init__(self, vector_store: WeaviateVectorStore):
        """
        Initialize workflow

        Args:
            vector_store: Weaviate vector store instance
        """
        self.vector_store = vector_store

        # Initialize agents
        self.input_validation_agent = InputValidationAgent()
        self.retrieval_agent = create_retrieval_agent(vector_store)
        self.calculation_agent = create_calculation_agent()
        self.synthesis_agent = create_synthesis_agent()

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create graph
        workflow = StateGraph(FinancialAnalysisState)

        # Add nodes
        workflow.add_node("input_validation", self.input_validation_agent)
        workflow.add_node("retrieval", self.retrieval_agent)
        workflow.add_node("calculation", self.calculation_agent)
        workflow.add_node("synthesis", self.synthesis_agent)

        # Define edges
        workflow.set_entry_point("input_validation")

        # Input validation → Retrieval (always)
        workflow.add_edge("input_validation", "retrieval")

        # Retrieval → Calculation or Synthesis (conditional)
        workflow.add_conditional_edges(
            "retrieval",
            self._should_calculate,
            {
                "calculate": "calculation",
                "synthesize": "synthesis"
            }
        )

        # Calculation → Synthesis
        workflow.add_edge("calculation", "synthesis")

        # Synthesis → END
        workflow.add_edge("synthesis", END)

        # Compile graph
        return workflow.compile()

    def _should_calculate(self, state: FinancialAnalysisState) -> Literal["calculate", "synthesize"]:
        """
        Determine if calculation is needed

        Args:
            state: Current state

        Returns:
            Next node to visit
        """
        intent = state.get("intent", "")

        if intent in ["calculation", "comparison"]:
            logger.info("Routing to calculation agent")
            return "calculate"
        else:
            logger.info("Skipping calculation, going to synthesis")
            return "synthesize"

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the workflow

        Args:
            question: User's question

        Returns:
            Final state with response
        """
        logger.info("=" * 80)
        logger.info(f"Processing query: {question}")
        logger.info("=" * 80)

        # Create initial state
        initial_state = create_initial_state(question)

        # Run workflow
        final_state = self.graph.invoke(initial_state)

        # Log completion
        logger.info("=" * 80)
        logger.info("Query processing complete")
        logger.info(f"Confidence: {final_state['confidence_score']:.3f}")
        logger.info(f"Agents used: {' → '.join(final_state['agent_sequence'])}")
        logger.info("=" * 80)

        return final_state

    def get_response(self, state: Dict[str, Any]) -> str:
        """
        Extract formatted response from state

        Args:
            state: Final state

        Returns:
            Formatted response string
        """
        synthesis = state.get("synthesis", "No response generated")
        confidence = state.get("confidence_score", 0.0)
        citations = state.get("citations", [])

        response = f"{synthesis}\n\n"

        # Add confidence indicator
        if confidence < 0.7:
            response += f"⚠️ Low Confidence ({confidence:.1%}) - Human review recommended\n\n"
        else:
            response += f"✓ Confidence: {confidence:.1%}\n\n"

        # Add citations
        if citations:
            response += "Sources:\n"
            for i, citation in enumerate(citations[:5], 1):  # Top 5
                response += (
                    f"{i}. {citation['source']}, Page {citation['page']} "
                    f"(Q{citation['fiscal_quarter']} {citation['fiscal_year']}) "
                    f"[Score: {citation['score']:.3f}]\n"
                )

        return response


def create_workflow(vector_store: WeaviateVectorStore) -> FinancialAnalysisWorkflow:
    """
    Factory function to create workflow

    Args:
        vector_store: Weaviate vector store

    Returns:
        FinancialAnalysisWorkflow instance
    """
    return FinancialAnalysisWorkflow(vector_store)
