"""
src/agent/response_parser.py
-----------------------------
Parses raw LLM text output into a validated pipeline specification dict.

Three-stage fallback strategy:
  Stage 1: Direct JSON parse of the full response string.
  Stage 2: Extract JSON from a markdown code block (```json ... ```).
  Stage 3: Brute-force scan for the outermost { ... } block.
  Fallback: Return None — caller must handle with a default spec.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parses raw LLM text into a pipeline specification dict.
    Returns None if all three parsing strategies fail.
    """

    def parse(self, raw: str) -> dict | None:
        """
        Args:
            raw: Raw string returned by the LLM API.

        Returns:
            Parsed dict or None if parsing fails entirely.
        """
        if not raw or not raw.strip():
            logger.error("Parser received empty or null response.")
            return None

        # ── Stage 1: Direct JSON parse ───────────────────────────────────────
        try:
            result = json.loads(raw.strip())
            logger.debug("Parser: Stage 1 (direct) succeeded.")
            return result
        except json.JSONDecodeError:
            pass

        # ── Stage 2: JSON inside markdown code fence ─────────────────────────
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw,
            re.DOTALL
        )
        if match:
            try:
                result = json.loads(match.group(1))
                logger.debug("Parser: Stage 2 (code fence) succeeded.")
                return result
            except json.JSONDecodeError:
                pass

        # ── Stage 3: Find outermost { ... } block ───────────────────────────
        # Handles cases where the LLM adds prose before/after the JSON
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                logger.debug("Parser: Stage 3 (brace scan) succeeded.")
                return result
            except json.JSONDecodeError:
                pass

        logger.error(
            f"ResponseParser: All three stages failed.\n"
            f"First 300 chars of response: {raw[:300]}"
        )
        return None
