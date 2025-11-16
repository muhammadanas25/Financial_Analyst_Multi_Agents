"""
Input Validation Agent
Responsibilities:
- Query sanitization
- Temporal context extraction (Q3 2024, FY2023)
- Intent classification (retrieval, calculation, comparison, analysis)
- Query rewriting for clarity
"""
import re
from typing import Dict, Any
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import FinancialAnalysisState, log_agent_action
from config.config import settings


class InputValidationAgent:
    """Agent for validating and preprocessing user queries"""

    def __init__(self):
        """Initialize input validation agent"""
        self.llm = ChatOpenAI(
            model=settings.fallback_llm_model,  # Use cheaper model for classification
            temperature=0
        )

        # Intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial query classifier. Classify the user's query into one of these intents:

- retrieval: Simple fact retrieval (e.g., "What was revenue in Q1 2025?")
- calculation: Requires numerical calculation (e.g., "Calculate the profit margin")
- comparison: Comparing across periods or entities (e.g., "Compare Q1 2025 vs Q1 2024")
- analysis: Deep analysis required (e.g., "Analyze the factors driving revenue growth")

Also determine query complexity:
- simple: Single fact retrieval
- multi-hop: Requires combining multiple pieces of information
- complex: Requires deep reasoning and synthesis

Respond in JSON format:
{{
    "intent": "retrieval|calculation|comparison|analysis",
    "complexity": "simple|multi-hop|complex",
    "reasoning": "brief explanation"
}}"""),
            ("user", "{query}")
        ])

    def __call__(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Process and validate input query

        Args:
            state: Current state

        Returns:
            Updated state with validated query and metadata
        """
        query = state["original_query"]
        logger.info(f"Input Validation Agent processing: {query}")

        # Extract temporal context
        temporal_context = self._extract_temporal_context(query)

        # Classify intent and complexity
        classification = self._classify_query(query)

        # Update state
        state["temporal_context"] = temporal_context
        state["intent"] = classification["intent"]
        state["query_complexity"] = classification["complexity"]

        # Log action
        state = log_agent_action(
            state,
            agent_name="InputValidationAgent",
            action="validate_and_classify",
            details={
                "temporal_context": temporal_context,
                "intent": classification["intent"],
                "complexity": classification["complexity"],
                "reasoning": classification.get("reasoning", ""),
            }
        )

        logger.info(
            f"Classified as: {classification['intent']} ({classification['complexity']}) "
            f"- Temporal: {temporal_context}"
        )

        return state

    def _extract_temporal_context(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from query"""
        context = {}

        # Fiscal year patterns
        year_patterns = [
            r'FY\s*(\d{4})',
            r'fiscal\s*year\s*(\d{4})',
            r'\b(20\d{2})\b',
        ]

        for pattern in year_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2000 <= year <= 2030:
                    context["fiscal_year"] = year
                    break

        # Fiscal quarter patterns
        quarter_patterns = [
            r'Q([1-4])\s*(\d{4})?',
            r'([1-4])Q\s*(\d{4})?',
            r'quarter\s*([1-4])',
        ]

        for pattern in quarter_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                quarter = int(match.group(1))
                context["fiscal_quarter"] = quarter

                # Extract year if present in match
                if len(match.groups()) > 1 and match.group(2):
                    year = int(match.group(2))
                    if 2000 <= year <= 2030:
                        context["fiscal_year"] = year
                break

        # Comparison patterns
        if re.search(r'\bvs\b|\bversus\b|\bcompare\b|\bcompared to\b', query, re.IGNORECASE):
            context["is_comparison"] = True

            # Try to extract both periods
            quarters = re.findall(r'Q([1-4])\s*(\d{4})', query, re.IGNORECASE)
            if len(quarters) >= 2:
                context["comparison_periods"] = [
                    {"quarter": int(q[0]), "year": int(q[1])}
                    for q in quarters[:2]
                ]

        # Time range patterns
        if re.search(r'last\s+(\d+)\s+(year|quarter|month)', query, re.IGNORECASE):
            match = re.search(r'last\s+(\d+)\s+(year|quarter|month)', query, re.IGNORECASE)
            context["time_range"] = {
                "count": int(match.group(1)),
                "unit": match.group(2).lower()
            }

        return context

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query intent and complexity using LLM"""
        try:
            # Create chain
            chain = self.intent_prompt | self.llm

            # Get classification
            response = chain.invoke({"query": query})

            # Parse JSON response
            import json
            result = json.loads(response.content)

            return result

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Fallback to heuristic classification
            return self._heuristic_classification(query)

    def _heuristic_classification(self, query: str) -> Dict[str, Any]:
        """Fallback heuristic classification if LLM fails"""
        query_lower = query.lower()

        # Determine intent
        if any(word in query_lower for word in ['calculate', 'compute', 'what is the', 'margin', 'ratio', 'rate']):
            intent = "calculation"
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference between']):
            intent = "comparison"
        elif any(word in query_lower for word in ['analyze', 'analysis', 'explain', 'why', 'how', 'factors', 'drivers']):
            intent = "analysis"
        else:
            intent = "retrieval"

        # Determine complexity
        if intent in ["comparison", "analysis"] or "?" in query and len(query.split()) > 15:
            complexity = "complex"
        elif any(word in query_lower for word in ['and', 'then', 'also', 'additionally']):
            complexity = "multi-hop"
        else:
            complexity = "simple"

        return {
            "intent": intent,
            "complexity": complexity,
            "reasoning": "Heuristic classification (LLM fallback)"
        }
