"""
LangGraph state schema for financial analysis workflow.
Defines the state that flows through the multi-agent system.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add


class FinancialAnalysisState(TypedDict):
    """
    State for financial analysis multi-agent workflow.

    This state flows through all agents and accumulates information
    at each step for complete audit trails.
    """

    # ========== Query Information ==========
    original_query: str  # User's original question
    intent: str  # Classification: retrieval, calculation, comparison, analysis
    temporal_context: Dict[str, Any]  # fiscal_year, quarter, date_range
    query_complexity: str  # simple, multi-hop, complex

    # ========== Retrieved Information ==========
    retrieved_documents: Annotated[List[Dict[str, Any]], add]  # Retrieved chunks
    document_metadata: List[Dict[str, Any]]  # Metadata for retrieved docs
    retrieval_score: float  # Average relevance score

    # ========== Calculations ==========
    calculations: Dict[str, Any]  # All numerical computations
    calculation_verification: Dict[str, Any]  # Secondary validation results
    calculation_errors: Annotated[List[str], add]  # Any calculation errors

    # ========== Analysis ==========
    analysis_results: Dict[str, Any]  # Analysis from analysis agent
    comparison_results: Dict[str, Any]  # Comparison results if applicable

    # ========== Validation ==========
    validation_passed: bool  # Overall validation status
    confidence_score: float  # Overall confidence (0-1)
    hallucination_flags: Annotated[List[Dict[str, Any]], add]  # Detected hallucinations
    calculation_verified: bool  # Calculation verification status

    # ========== Response ==========
    synthesis: str  # Final synthesized answer
    citations: List[Dict[str, Any]]  # Source citations
    reasoning_steps: Annotated[List[str], add]  # Step-by-step reasoning

    # ========== Audit Trail ==========
    agent_sequence: Annotated[List[str], add]  # Sequence of agents called
    audit_log: Annotated[List[Dict[str, Any]], add]  # Complete audit log
    timestamps: Dict[str, str]  # Timestamps for each step

    # ========== Flags ==========
    needs_human_review: bool  # Flag for human-in-the-loop
    error_message: Optional[str]  # Error message if any


def create_initial_state(query: str) -> FinancialAnalysisState:
    """
    Create initial state for a new query

    Args:
        query: User's question

    Returns:
        Initial state dictionary
    """
    from datetime import datetime

    return {
        # Query information
        "original_query": query,
        "intent": "",
        "temporal_context": {},
        "query_complexity": "",

        # Retrieved information
        "retrieved_documents": [],
        "document_metadata": [],
        "retrieval_score": 0.0,

        # Calculations
        "calculations": {},
        "calculation_verification": {},
        "calculation_errors": [],

        # Analysis
        "analysis_results": {},
        "comparison_results": {},

        # Validation
        "validation_passed": False,
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "calculation_verified": False,

        # Response
        "synthesis": "",
        "citations": [],
        "reasoning_steps": [],

        # Audit trail
        "agent_sequence": [],
        "audit_log": [],
        "timestamps": {"created": datetime.now().isoformat()},

        # Flags
        "needs_human_review": False,
        "error_message": None,
    }


def log_agent_action(
    state: FinancialAnalysisState,
    agent_name: str,
    action: str,
    details: Dict[str, Any]
) -> FinancialAnalysisState:
    """
    Log an agent action to the audit trail

    Args:
        state: Current state
        agent_name: Name of the agent
        action: Action taken
        details: Additional details

    Returns:
        Updated state
    """
    from datetime import datetime

    # Add to agent sequence
    state["agent_sequence"].append(agent_name)

    # Add to audit log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "details": details,
    }
    state["audit_log"].append(log_entry)

    return state
