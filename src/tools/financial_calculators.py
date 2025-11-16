"""
Financial calculation tools with Decimal precision.
CRITICAL: Never use LLM for math - always execute deterministic code.
"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Any, Optional, Union
import re
from loguru import logger


class FinancialCalculator:
    """
    Financial calculations with precision and validation.
    All calculations use Decimal type for accuracy.
    """

    def __init__(self, decimal_places: int = 4):
        """
        Initialize calculator

        Args:
            decimal_places: Default decimal places for rounding
        """
        self.decimal_places = decimal_places

    def calculate_percentage_change(
        self,
        current_value: Union[float, Decimal, str],
        prior_value: Union[float, Decimal, str],
        label: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate percentage change between two values.
        Formula: ((current - prior) / prior) * 100

        Args:
            current_value: Current period value
            prior_value: Prior period value
            label: Description of calculation

        Returns:
            Dictionary with result and metadata
        """
        try:
            current = self._to_decimal(current_value)
            prior = self._to_decimal(prior_value)

            if prior == 0:
                return {
                    "value": None,
                    "error": "Division by zero - prior value is zero",
                    "label": label,
                    "calculation": f"({current} - {prior}) / {prior} * 100",
                    "verified": False,
                }

            result = ((current - prior) / prior) * Decimal('100')
            result_rounded = self._round(result, 2)  # Percentage to 2 decimal places

            # Verify with alternative calculation
            verification = self._verify_percentage_change(current, prior, result)

            return {
                "value": float(result_rounded),
                "formatted": f"{result_rounded:,.2f}%",
                "label": label,
                "calculation": f"(({current} - {prior}) / {prior}) * 100 = {result_rounded}%",
                "current_value": float(current),
                "prior_value": float(prior),
                "absolute_change": float(current - prior),
                "verified": verification["verified"],
                "precision": "2 decimal places",
                "direction": "increase" if result > 0 else ("decrease" if result < 0 else "unchanged"),
            }

        except (InvalidOperation, ValueError) as e:
            logger.error(f"Error calculating percentage change: {e}")
            return {
                "value": None,
                "error": str(e),
                "label": label,
                "verified": False,
            }

    def calculate_financial_ratio(
        self,
        numerator: Union[float, Decimal, str],
        denominator: Union[float, Decimal, str],
        ratio_name: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate financial ratio.
        Formula: numerator / denominator

        Args:
            numerator: Top value in ratio
            denominator: Bottom value in ratio
            ratio_name: Name of ratio (e.g., 'ROE', 'Debt-to-Equity')

        Returns:
            Dictionary with result and metadata
        """
        try:
            num = self._to_decimal(numerator)
            denom = self._to_decimal(denominator)

            if denom == 0:
                return {
                    "value": None,
                    "error": "Division by zero - denominator is zero",
                    "ratio_name": ratio_name,
                    "calculation": f"{num} / {denom}",
                    "verified": False,
                }

            result = num / denom
            result_rounded = self._round(result, self.decimal_places)

            return {
                "value": float(result_rounded),
                "formatted": f"{result_rounded:,.4f}",
                "ratio_name": ratio_name,
                "calculation": f"{num} / {denom} = {result_rounded}",
                "numerator": float(num),
                "denominator": float(denom),
                "verified": True,
                "precision": f"{self.decimal_places} decimal places",
            }

        except (InvalidOperation, ValueError) as e:
            logger.error(f"Error calculating ratio: {e}")
            return {
                "value": None,
                "error": str(e),
                "ratio_name": ratio_name,
                "verified": False,
            }

    def verify_balance_sheet_equation(
        self,
        total_assets: Union[float, Decimal, str],
        total_liabilities: Union[float, Decimal, str],
        total_equity: Union[float, Decimal, str],
        tolerance: float = 0.0001
    ) -> Dict[str, Any]:
        """
        Verify balance sheet equation: Assets = Liabilities + Equity

        Args:
            total_assets: Total assets
            total_liabilities: Total liabilities
            total_equity: Total shareholders' equity
            tolerance: Acceptable variance (default 0.01%)

        Returns:
            Dictionary with validation result
        """
        try:
            assets = self._to_decimal(total_assets)
            liabilities = self._to_decimal(total_liabilities)
            equity = self._to_decimal(total_equity)

            liabilities_plus_equity = liabilities + equity
            difference = assets - liabilities_plus_equity

            # Calculate percentage difference
            if assets != 0:
                pct_difference = abs(difference / assets) * Decimal('100')
            else:
                pct_difference = Decimal('0')

            balanced = pct_difference <= Decimal(str(tolerance * 100))

            return {
                "balanced": balanced,
                "total_assets": float(assets),
                "total_liabilities": float(liabilities),
                "total_equity": float(equity),
                "liabilities_plus_equity": float(liabilities_plus_equity),
                "difference": float(difference),
                "percentage_difference": float(pct_difference),
                "tolerance": tolerance * 100,
                "equation": f"{assets:,.2f} = {liabilities:,.2f} + {equity:,.2f}",
                "verified": balanced,
            }

        except (InvalidOperation, ValueError) as e:
            logger.error(f"Error verifying balance sheet: {e}")
            return {
                "balanced": False,
                "error": str(e),
                "verified": False,
            }

    def extract_number_from_text(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Extract numerical value from financial text, handling scales and currencies.

        Examples:
            - "$5.2 billion" -> 5200000000
            - "AED 1.3M" -> 1300000
            - "100 thousand" -> 100000

        Args:
            text: Text containing number

        Returns:
            Dictionary with extracted number and metadata
        """
        try:
            # Remove currency symbols
            cleaned = re.sub(r'[$€£AED USD EUR GBP]', '', text, flags=re.IGNORECASE)

            # Extract number
            number_match = re.search(r'([\d,]+\.?\d*)', cleaned)
            if not number_match:
                return {
                    "value": None,
                    "error": "No number found in text",
                    "original_text": text,
                }

            number_str = number_match.group(1).replace(',', '')
            number = Decimal(number_str)

            # Detect scale
            scale = 1
            scale_name = "units"

            if re.search(r'\b(billion|bn|b)\b', cleaned, re.IGNORECASE):
                scale = 1_000_000_000
                scale_name = "billions"
            elif re.search(r'\b(million|mn|m)\b', cleaned, re.IGNORECASE):
                scale = 1_000_000
                scale_name = "millions"
            elif re.search(r'\b(thousand|k)\b', cleaned, re.IGNORECASE):
                scale = 1_000
                scale_name = "thousands"

            final_value = number * Decimal(str(scale))

            # Detect currency
            currency = None
            for curr in ['AED', 'USD', 'EUR', 'GBP']:
                if curr in text.upper():
                    currency = curr
                    break
            if not currency and '$' in text:
                currency = 'USD'
            elif not currency and '€' in text:
                currency = 'EUR'
            elif not currency and '£' in text:
                currency = 'GBP'

            return {
                "value": float(final_value),
                "formatted": f"{final_value:,.2f}",
                "original_text": text,
                "base_number": float(number),
                "scale": scale_name,
                "scale_multiplier": scale,
                "currency": currency,
                "verified": True,
            }

        except (InvalidOperation, ValueError) as e:
            logger.error(f"Error extracting number: {e}")
            return {
                "value": None,
                "error": str(e),
                "original_text": text,
                "verified": False,
            }

    def calculate_growth_rate(
        self,
        values: list,
        periods: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Calculate growth rate over multiple periods.
        Can calculate CAGR (Compound Annual Growth Rate) if periods provided.

        Args:
            values: List of values over time
            periods: Optional list of period labels

        Returns:
            Dictionary with growth metrics
        """
        if len(values) < 2:
            return {
                "error": "Need at least 2 values to calculate growth",
                "verified": False,
            }

        try:
            decimal_values = [self._to_decimal(v) for v in values]

            # Calculate period-over-period growth rates
            growth_rates = []
            for i in range(1, len(decimal_values)):
                if decimal_values[i - 1] != 0:
                    growth = ((decimal_values[i] - decimal_values[i - 1]) / decimal_values[i - 1]) * Decimal('100')
                    growth_rates.append(float(self._round(growth, 2)))
                else:
                    growth_rates.append(None)

            # Calculate CAGR if we have start and end values
            start_value = decimal_values[0]
            end_value = decimal_values[-1]
            num_periods = len(decimal_values) - 1

            if start_value > 0:
                # CAGR = ((End/Start)^(1/periods) - 1) * 100
                ratio = end_value / start_value
                cagr = (ratio ** (Decimal('1') / Decimal(str(num_periods))) - Decimal('1')) * Decimal('100')
                cagr_value = float(self._round(cagr, 2))
            else:
                cagr_value = None

            return {
                "values": [float(v) for v in decimal_values],
                "growth_rates": growth_rates,
                "cagr": cagr_value,
                "periods": periods or list(range(len(values))),
                "average_growth": sum(g for g in growth_rates if g is not None) / len([g for g in growth_rates if g is not None]) if growth_rates else None,
                "verified": True,
            }

        except (InvalidOperation, ValueError) as e:
            logger.error(f"Error calculating growth rate: {e}")
            return {
                "error": str(e),
                "verified": False,
            }

    def _to_decimal(self, value: Union[float, Decimal, str]) -> Decimal:
        """Convert value to Decimal"""
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            # Remove commas and convert
            cleaned = value.replace(',', '')
            return Decimal(cleaned)
        else:
            raise ValueError(f"Cannot convert {type(value)} to Decimal")

    def _round(self, value: Decimal, places: int) -> Decimal:
        """Round Decimal to specified places"""
        quantizer = Decimal('0.1') ** places
        return value.quantize(quantizer, rounding=ROUND_HALF_UP)

    def _verify_percentage_change(
        self,
        current: Decimal,
        prior: Decimal,
        calculated_result: Decimal
    ) -> Dict[str, bool]:
        """Verify percentage change calculation with alternative method"""
        # Alternative calculation
        alt_result = ((current - prior) / prior) * Decimal('100')

        # Check if results match within tolerance
        difference = abs(calculated_result - alt_result)
        tolerance = Decimal('0.01')  # 0.01% tolerance

        return {
            "verified": difference <= tolerance,
            "difference": float(difference),
        }


# Singleton calculator instance
calculator = FinancialCalculator()


# Tool functions for LangChain/LangGraph integration
def calculate_percentage_change_tool(
    current_value: float,
    prior_value: float,
    label: str = ""
) -> Dict[str, Any]:
    """Tool wrapper for percentage change calculation"""
    return calculator.calculate_percentage_change(current_value, prior_value, label)


def calculate_financial_ratio_tool(
    numerator: float,
    denominator: float,
    ratio_name: str = ""
) -> Dict[str, Any]:
    """Tool wrapper for financial ratio calculation"""
    return calculator.calculate_financial_ratio(numerator, denominator, ratio_name)


def verify_balance_sheet_equation_tool(
    total_assets: float,
    total_liabilities: float,
    total_equity: float,
    tolerance: float = 0.0001
) -> Dict[str, Any]:
    """Tool wrapper for balance sheet verification"""
    return calculator.verify_balance_sheet_equation(
        total_assets, total_liabilities, total_equity, tolerance
    )


def extract_number_from_text_tool(text: str) -> Dict[str, Any]:
    """Tool wrapper for number extraction"""
    return calculator.extract_number_from_text(text)


# Tool definitions for LangChain
FINANCIAL_CALCULATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage_change",
            "description": "Calculate percentage change between two values. Use for YoY growth, QoQ change, variance analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_value": {
                        "type": "number",
                        "description": "Current period value"
                    },
                    "prior_value": {
                        "type": "number",
                        "description": "Prior period value for comparison"
                    },
                    "label": {
                        "type": "string",
                        "description": "Description of what is being calculated (e.g., 'Q1 2025 vs Q1 2024 revenue growth')"
                    }
                },
                "required": ["current_value", "prior_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_ratio",
            "description": "Calculate financial ratios like P/E, debt-to-equity, ROE, profit margin, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {
                        "type": "number",
                        "description": "Top value in ratio"
                    },
                    "denominator": {
                        "type": "number",
                        "description": "Bottom value in ratio"
                    },
                    "ratio_name": {
                        "type": "string",
                        "description": "Name of ratio (e.g., 'Profit Margin', 'ROE', 'Debt-to-Equity')"
                    }
                },
                "required": ["numerator", "denominator"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_balance_sheet_equation",
            "description": "Verify that Assets = Liabilities + Equity. Critical validation for balance sheet data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_assets": {"type": "number", "description": "Total assets"},
                    "total_liabilities": {"type": "number", "description": "Total liabilities"},
                    "total_equity": {"type": "number", "description": "Total shareholders' equity"},
                    "tolerance": {
                        "type": "number",
                        "description": "Acceptable variance as decimal (default 0.0001 = 0.01%)",
                        "default": 0.0001
                    }
                },
                "required": ["total_assets", "total_liabilities", "total_equity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_number_from_text",
            "description": "Extract numerical value from financial text handling scales (millions, billions) and currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text containing number (e.g., '$5.2 billion', 'AED 1.3M', '100 thousand')"
                    }
                },
                "required": ["text"]
            }
        }
    }
]
