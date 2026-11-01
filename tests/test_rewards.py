"""Comprehensive test suite for Vexel GRPO deterministic reward functions."""

from __future__ import annotations

import json
import pytest
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

from vexel_rewards import (
    conciseness_latency_reward,
    directional_alpha_reward,
    pydantic_schema_compliance_reward,
    strict_json_parse_reward,
)


class SampleAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SimpleSignalSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    action: SampleAction
    confidence: float = Field(..., ge=0.0, le=1.0)


class TestStrictJsonParseReward:
    """Verify raw JSON parsing vs markdown wrapping penalties."""

    def test_pure_raw_json(self):
        prompts = ["Analyze signal"]
        completions = ['{"ticker": "BTC", "action": "LONG", "confidence": 0.95}']
        rewards = strict_json_parse_reward(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_markdown_wrapped_json(self):
        prompts = ["Analyze signal"]
        completions = [
            '```json\n{"ticker": "BTC", "action": "LONG", "confidence": 0.95}\n```'
        ]
        rewards = strict_json_parse_reward(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == 0.2

    def test_malformed_json(self):
        prompts = ["Analyze signal"]
        completions = [
            '{"ticker": "BTC", "action": "LONG", confidence: 0.95'  # missing brace and quote
        ]
        rewards = strict_json_parse_reward(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == -1.0

    def test_conversational_text_no_json(self):
        prompts = ["Analyze signal"]
        completions = ["I think Bitcoin will go up due to ETF inflows."]
        rewards = strict_json_parse_reward(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == -1.0

    def test_chat_format_completions(self):
        prompts = ["Analyze signal"]
        completions = [
            [{"role": "assistant", "content": '{"ticker": "SOL", "action": "LONG", "confidence": 0.88}'}]
        ]
        rewards = strict_json_parse_reward(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == 1.0


class TestPydanticSchemaComplianceReward:
    """Verify key presence, extra key penalties, and bounds enforcement."""

    def test_perfect_compliance(self):
        reward_fn = pydantic_schema_compliance_reward(SimpleSignalSchema)
        prompts = ["Analyze signal"]
        completions = ['{"ticker": "ETH", "action": "LONG", "confidence": 0.9}']
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_hallucinated_extra_key(self):
        reward_fn = pydantic_schema_compliance_reward(SimpleSignalSchema, strict_extra_penalty=0.3)
        prompts = ["Analyze signal"]
        # Extra hallucinated key: 'reasoning_thought'
        completions = [
            '{"ticker": "ETH", "action": "LONG", "confidence": 0.9, "reasoning_thought": "Bullish news."}'
        ]
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        # Should be penalized for extra key
        assert rewards[0] < 1.0
        assert rewards[0] == pytest.approx(0.7, 0.05)

    def test_missing_required_key(self):
        reward_fn = pydantic_schema_compliance_reward(SimpleSignalSchema, missing_key_penalty=0.2)
        prompts = ["Analyze signal"]
        # Missing 'confidence' key
        completions = ['{"ticker": "ETH", "action": "LONG"}']
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] < 0.5

    def test_invalid_enum_value(self):
        reward_fn = pydantic_schema_compliance_reward(SimpleSignalSchema)
        prompts = ["Analyze signal"]
        # 'HOLD' is not a valid SampleAction (must be LONG, SHORT, NEUTRAL)
        completions = ['{"ticker": "ETH", "action": "HOLD", "confidence": 0.9}']
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] < 0.0

    def test_out_of_bounds_numerical_field(self):
        reward_fn = pydantic_schema_compliance_reward(SimpleSignalSchema)
        prompts = ["Analyze signal"]
        # confidence 1.5 > 1.0
        completions = ['{"ticker": "ETH", "action": "LONG", "confidence": 1.5}']
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] < 0.0


class TestConcisenessLatencyReward:
    """Verify exponential penalty on verbose chain-of-thought commentary."""

    def test_compact_completion(self):
        reward_fn = conciseness_latency_reward(char_penalty_threshold=100)
        prompts = ["Prompt"]
        completions = ['{"ticker": "BTC", "action": "LONG"}']  # len ~35 chars
        rewards = reward_fn(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_bloated_cot_completion(self):
        reward_fn = conciseness_latency_reward(char_penalty_threshold=80, decay_lambda=0.02)
        prompts = ["Prompt"]
        # Long 350-character chain-of-thought text
        long_cot = (
            "Let's think step by step. First, we examine the macroeconomic data. "
            "The Federal Reserve cut rates by 50 basis points. This increases dollar liquidity "
            "and promotes risk-on assets such as Bitcoin and Ethereum. Therefore: "
            '{"ticker": "BTC", "action": "LONG", "confidence": 0.95}'
        )
        rewards = reward_fn(prompts, completions=[long_cot])
        assert len(rewards) == 1
        assert rewards[0] < 0.0  # severely penalized


class TestDirectionalAlphaReward:
    """Verify trading alpha evaluation against ground truth."""

    def test_perfect_trading_payload(self):
        prompts = ["Flash news"]
        completions = ['{"ticker": "SOL", "direction_bias": "LONG", "confidence": 0.95}']
        gt = [{"ticker": "SOL", "direction_bias": "LONG", "confidence": 0.95}]
        rewards = directional_alpha_reward(prompts, completions, ground_truth=gt)
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(1.0, 0.05)

    def test_opposite_direction_penalty(self):
        prompts = ["Flash news"]
        # Model predicted SHORT, but ground truth is LONG
        completions = ['{"ticker": "SOL", "direction_bias": "SHORT", "confidence": 0.95}']
        gt = [{"ticker": "SOL", "direction_bias": "LONG", "confidence": 0.95}]
        rewards = directional_alpha_reward(prompts, completions, ground_truth=gt)
        assert len(rewards) == 1
        # Direction penalty (-0.4) reduces total reward significantly
        assert rewards[0] < 0.3

    def test_proximity_delta_prob(self):
        prompts = ["Prediction market"]
        completions = ['{"market_id": "test", "execution_action": "BUY_YES", "delta_yes_prob": 0.22}']
        gt = [{"market_id": "test", "execution_action": "BUY_YES", "delta_yes_prob": 0.20}]
        rewards = directional_alpha_reward(prompts, completions, ground_truth=gt)
        assert len(rewards) == 1
        assert rewards[0] > 0.8
