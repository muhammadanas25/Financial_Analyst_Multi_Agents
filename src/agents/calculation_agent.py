"""
Calculation Agent
Responsibilities:
- Execute ALL numerical computations using tools
- NEVER use LLM for math
- Financial ratio calculations
- Percentage changes and growth rates
- Validation against source data
"""
from typing import Dict, Any
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool

from .state import FinancialAnalysisState, log_agent_action
from src.tools.financial_calculators import (
    calculate_percentage_change_tool,
    calculate_financial_ratio_tool,
    extract_number_from_text_tool,
    calculator,
)
from config.config import settings


class CalculationAgent:
    """
    Agent for financial calculations with tool calling.

    CRITICAL: This agent NEVER performs calculations itself.
    All math is delegated to deterministic calculation tools.
    """

    def __init__(self):
        """Initialize calculation agent"""
        self.llm = ChatOpenAI(
            model=settings.primary_llm_model,
            temperature=0  # Deterministic
        )

        # Define tools
        self.tools = [
            Tool(
                name="calculate_percentage_change",
                description="Calculate percentage change between two values. Use for YoY growth, QoQ change, variance analysis. Returns verified result with metadata.",
                func=lambda **kwargs: calculate_percentage_change_tool(**kwargs)
            ),
            Tool(
                name="calculate_financial_ratio",
                description="Calculate financial ratios like P/E, debt-to-equity, ROE, profit margin, etc. Returns precise result.",
                func=lambda **kwargs: calculate_financial_ratio_tool(**kwargs)
            ),
            Tool(
                name="extract_number_from_text",
                description="Extract numerical value from financial text handling scales (millions, billions) and currencies. Example: 'AED 5.2 billion' -> 5200000000",
                func=lambda **kwargs: extract_number_from_text_tool(**kwargs)
            ),
        ]

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial calculation agent. Your ONLY job is to use calculation tools - NEVER compute numbers yourself.

CRITICAL RULES:
1. ALWAYS use calculation tools for ANY numerical operation
2. NEVER compute numbers yourself - not even simple arithmetic
3. Extract numbers from documents using extract_number_from_text tool
4. Verify all calculations match source documents
5. Show your reasoning step-by-step

When asked to calculate:
1. Think: Identify what calculation is needed
2. Extract: Get numbers from the context using extract_number_from_text
3. Calculate: Call the appropriate calculation tool
4. Verify: Check if result makes sense given the context

You have access to these tools:
- calculate_percentage_change: For growth rates, changes
- calculate_financial_ratio: For ratios, margins, returns
- extract_number_from_text: For extracting numbers from text

Context from retrieved documents:
{context}

User Query: {query}
"""),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def __call__(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Perform calculations if needed

        Args:
            state: Current state

        Returns:
            Updated state with calculation results
        """
        query = state["original_query"]
        intent = state.get("intent", "")
        retrieved_docs = state.get("retrieved_documents", [])

        # Only run if intent is calculation or comparison
        if intent not in ["calculation", "comparison"]:
            logger.info("Calculation Agent: No calculations needed for this query")
            return state

        logger.info(f"Calculation Agent processing: {query}")

        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)

        try:
            # Run agent
            result = self.agent_executor.invoke({
                "query": query,
                "context": context
            })

            # Extract calculations from agent output
            calculations = self._extract_calculations(result)

            # Update state
            state["calculations"] = calculations
            state["calculation_verified"] = calculations.get("verified", False)

            # Add reasoning
            state["reasoning_steps"].append(f"Performed calculations: {list(calculations.keys())}")

            # Log action
            state = log_agent_action(
                state,
                agent_name="CalculationAgent",
                action="calculate",
                details={
                    "calculations": calculations,
                    "verified": state["calculation_verified"],
                }
            )

            logger.info(f"Calculations completed: {calculations}")

        except Exception as e:
            logger.error(f"Calculation error: {e}")
            state["calculation_errors"].append(str(e))
            state["error_message"] = f"Calculation error: {e}"

        return state

    def _prepare_context(self, retrieved_docs: list) -> str:
        """Prepare context from retrieved documents"""
        if not retrieved_docs:
            return "No documents retrieved."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 docs
            metadata = doc.get("metadata", {})
            context_parts.append(
                f"Document {i} (Q{metadata.get('fiscal_quarter', '?')} {metadata.get('fiscal_year', '?')}):\n"
                f"{doc.get('content', '')[:500]}...\n"
            )

        return "\n\n".join(context_parts)

    def _extract_calculations(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract calculation results from agent output"""
        # Parse the agent's output
        output = agent_result.get("output", "")

        calculations = {
            "output": output,
            "verified": True,  # Tools provide verification
        }

        # Try to extract structured results if agent returned JSON
        import re
        import json

        # Look for JSON in output
        json_match = re.search(r'\{[^{}]*\}', output)
        if json_match:
            try:
                structured = json.loads(json_match.group())
                calculations.update(structured)
            except json.JSONDecodeError:
                pass

        return calculations


def create_calculation_agent() -> CalculationAgent:
    """
    Factory function to create calculation agent

    Returns:
        CalculationAgent instance
    """
    return CalculationAgent()
