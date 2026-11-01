"""Deterministic GRPO reward functions for strict schema adherence and low-latency SLMs."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError


def _extract_completion_text(completion: Any) -> str:
    """Extract raw string content from TRL completion representation.

    Supports:
    - Raw string (standard completions)
    - List of dicts (OpenAI/HuggingFace chat format `[{'role': 'assistant', 'content': '...'}]`)
    - Dict with 'content' or 'text' key
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[-1], dict):
            return str(completion[-1].get("content") or completion[-1].get("text", ""))
        return "".join(str(c) for c in completion)
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text", ""))
    return str(completion)


def _parse_json_with_metadata(text: str) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Inspect string for valid JSON and detect markdown codeblock wrappers.

    Returns:
        (parsed_dict, is_valid_json, has_markdown_fences)
    """
    clean = text.strip()
    if not clean:
        return None, False, False

    has_markdown = "```" in clean

    # Attempt 1: Raw JSON parsing
    if not has_markdown:
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed, True, False
        except Exception:
            pass

    # Attempt 2: Extract inside ```json ... ``` markdown fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        snippet = match.group(1).strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed, True, True
        except Exception:
            pass

    # Attempt 3: Outermost braces extraction
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        snippet = clean[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed, True, True
        except Exception:
            pass

    return None, False, has_markdown


def strict_json_parse_reward(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """TRL-compatible reward function scoring JSON purity and markdown wrapping.

    Scoring:
    - Raw valid JSON with ZERO markdown fences: +1.0
    - Valid JSON wrapped in markdown fences:     +0.2
    - Malformed / unparseable JSON:              -1.0
    """
    rewards: List[float] = []
    for comp in completions:
        text = _extract_completion_text(comp)
        parsed, is_valid, has_markdown = _parse_json_with_metadata(text)
        if is_valid and not has_markdown:
            rewards.append(1.0)
        elif is_valid and has_markdown:
            rewards.append(0.2)
        else:
            rewards.append(-1.0)
    return rewards


def pydantic_schema_compliance_reward(
    schema_cls: Type[BaseModel],
    strict_extra_penalty: float = 0.3,
    missing_key_penalty: float = 0.2,
) -> Callable[..., List[float]]:
    """Factory returning a TRL reward function enforcing strict Pydantic v2 adherence.

    Scoring breakdown:
    - Non-JSON: -1.0
    - Missing required fields: -missing_key_penalty per field
    - Hallucinated extra fields: -strict_extra_penalty per extra field
    - Valid Pydantic model with zero extra/missing fields: +1.0
    - Bounded in [-1.0, 1.0]
    """
    expected_fields = set(schema_cls.model_fields.keys())

    def reward_fn(
        prompts: List[Any],
        completions: List[Any],
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []
        for comp in completions:
            text = _extract_completion_text(comp)
            parsed, is_valid, _ = _parse_json_with_metadata(text)
            if not is_valid or parsed is None:
                rewards.append(-1.0)
                continue

            actual_keys = set(parsed.keys())
            missing_keys = expected_fields - actual_keys
            extra_keys = actual_keys - expected_fields

            # Evaluate core fields against schema
            core_payload = {k: v for k, v in parsed.items() if k in expected_fields}
            core_valid = False
            validation_errors = 0

            try:
                schema_cls.model_validate(core_payload)
                core_valid = True
            except ValidationError as ve:
                validation_errors = len(ve.errors())
            except Exception:
                validation_errors = 1

            if core_valid:
                # Core schema is fully valid; penalize strictly for missing and extra keys
                score = 1.0 - (len(extra_keys) * strict_extra_penalty) - (len(missing_keys) * missing_key_penalty)
                rewards.append(max(-1.0, min(1.0, score)))
            else:
                # Core schema has type/enum/bounds errors
                base_score = 0.1 - (0.25 * validation_errors) - (len(extra_keys) * strict_extra_penalty) - (len(missing_keys) * missing_key_penalty)
                rewards.append(max(-1.0, min(1.0, base_score)))

        return rewards

    reward_fn.__name__ = f"pydantic_schema_compliance_reward_{schema_cls.__name__}"
    return reward_fn


def conciseness_latency_reward(
    max_tokens_target: int = 64,
    char_penalty_threshold: int = 256,
    decay_lambda: float = 0.008,
) -> Callable[..., List[float]]:
    """Penalize chain-of-thought bloat for low-latency SLM deployment.

    Any commentary outside the pure JSON payload increases token count, TTFT,
    and end-to-end execution latency.

    Scoring:
    - len(completion) <= char_penalty_threshold: +1.0
    - len(completion) > char_penalty_threshold: smooth exponential decay
      reward = 2.0 * exp(-lambda * (len - threshold)) - 1.0 (smoothly drops to -1.0)
    """
    def reward_fn(
        prompts: List[Any],
        completions: List[Any],
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []
        for comp in completions:
            text = _extract_completion_text(comp).strip()
            char_len = len(text)
            if char_len <= char_penalty_threshold:
                rewards.append(1.0)
            else:
                excess = char_len - char_penalty_threshold
                decay = 2.0 * math.exp(-decay_lambda * excess) - 1.0
                rewards.append(max(-1.0, min(1.0, decay)))
        return rewards

    reward_fn.__name__ = f"conciseness_latency_reward_target_{max_tokens_target}"
    return reward_fn


def directional_alpha_reward(
    prompts: List[Any],
    completions: List[Any],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> List[float]:
    """Evaluate quantitative prediction accuracy against ground-truth labels.

    Evaluates:
    - Asset Ticker match: +0.3 (exact match)
    - Directional Bias / Execution Action match: +0.4 (exact match), -0.4 (inverted opposite action)
    - Numerical Proximity (probability delta, urgency, confidence): up to +0.3
    - Total reward in [-1.0, 1.0]
    """
    # Attempt to locate ground_truth in kwargs or metadata
    gt_list = ground_truth or kwargs.get("ground_truth") or kwargs.get("expected")
    rewards: List[float] = []

    for idx, comp in enumerate(completions):
        text = _extract_completion_text(comp)
        parsed, is_valid, _ = _parse_json_with_metadata(text)
        if not is_valid or parsed is None:
            rewards.append(-1.0)
            continue

        gt: Optional[Dict[str, Any]] = None
        if gt_list and idx < len(gt_list):
            gt = gt_list[idx]

        if not gt:
            # If no ground truth available, award neutral score if valid JSON
            rewards.append(0.5)
            continue

        score = 0.0

        # 1. Identifier match (ticker or market_id) (+0.3)
        id_key = "ticker" if "ticker" in gt else ("market_id" if "market_id" in gt else None)
        if id_key and id_key in parsed:
            if str(parsed[id_key]).strip().upper() == str(gt[id_key]).strip().upper():
                score += 0.3
            else:
                score -= 0.1

        # 2. Directional Bias or Execution Action (+0.4)
        pred_dir = str(parsed.get("direction_bias") or parsed.get("execution_action") or "").upper()
        true_dir = str(gt.get("direction_bias") or gt.get("execution_action") or "").upper()

        if pred_dir and true_dir:
            if pred_dir == true_dir:
                score += 0.4
            elif (
                (pred_dir == "LONG" and true_dir == "SHORT")
                or (pred_dir == "SHORT" and true_dir == "LONG")
                or (pred_dir == "BUY_YES" and true_dir == "BUY_NO")
                or (pred_dir == "BUY_NO" and true_dir == "BUY_YES")
            ):
                score -= 0.4
            else:
                score += 0.0  # NEUTRAL vs LONG/SHORT

        # 3. Numerical Proximity (+0.3)
        num_keys = ["delta_yes_prob", "urgency_score", "confidence"]
        matched_num_scores = []
        for k in num_keys:
            if k in gt and k in parsed:
                try:
                    p_val = float(parsed[k])
                    t_val = float(gt[k])
                    if k == "delta_yes_prob":
                        # Range [-1.0, 1.0], max diff is 2.0
                        diff = abs(p_val - t_val)
                        sub_score = max(0.0, 1.0 - (diff / 2.0))
                    else:
                        # Range [0.0, 1.0], max diff is 1.0
                        diff = abs(p_val - t_val)
                        sub_score = max(0.0, 1.0 - diff)
                    matched_num_scores.append(sub_score)
                except (ValueError, TypeError):
                    matched_num_scores.append(0.0)

        if matched_num_scores:
            avg_num = sum(matched_num_scores) / len(matched_num_scores)
            score += 0.3 * avg_num
        else:
            # If no numerical keys, distribute weight to ticker/direction
            score += 0.3 if score > 0 else 0.0

        rewards.append(max(-1.0, min(1.0, score)))

    return rewards
