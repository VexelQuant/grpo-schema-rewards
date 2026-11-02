"""Real integration script: Simulates trl.GRPOTrainer reward evaluation & GRPO advantage calculation."""

from __future__ import annotations

import json
import numpy as np
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

from vexel_rewards import (
    strict_json_parse_reward,
    pydantic_schema_compliance_reward,
    conciseness_latency_reward,
    directional_alpha_reward,
)


class CatalystType(str, Enum):
    REGULATORY_RULING = "REGULATORY_RULING"
    FED_RATE_DECISION = "FED_RATE_DECISION"
    TOKENOMICS_UPGRADE = "TOKENOMICS_UPGRADE"


class DirectionBias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class CatalystSignalSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(..., min_length=1, max_length=10)
    catalyst_type: CatalystType
    direction_bias: DirectionBias
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_60char_max: str = Field(..., max_length=60)


def main():
    print("=" * 80)
    print(" VexelQuant GRPO Reward Integration & Advantage Audit")
    print("=" * 80)

    # 1. Prepare target ground truth and candidate completions representing a generation group (G=6)
    ground_truth = {
        "ticker": "ETH",
        "catalyst_type": "REGULATORY_RULING",
        "direction_bias": "LONG",
        "urgency_score": 0.95,
        "confidence": 0.98,
        "reasoning_60char_max": "Spot ETH options approved by SEC; institutional inflows.",
    }
    prompts = [
        "ALERT: SEC approves BlackRock Spot ETH options on Nasdaq."
    ] * 6

    completions = [
        # Candidate 0: Perfect raw JSON, concise, exact alpha match
        json.dumps(ground_truth),

        # Candidate 1: Valid JSON but markdown-wrapped with backticks (```json)
        f"```json\n{json.dumps(ground_truth)}\n```",

        # Candidate 2: Valid JSON but hallucinated extra key ('internal_thought')
        json.dumps({**ground_truth, "internal_thought": "Bullish ETF flows expected"}),

        # Candidate 3: Valid JSON but inverted direction bias (SHORT instead of LONG)
        json.dumps({**ground_truth, "direction_bias": "SHORT"}),

        # Candidate 4: Verbose chain-of-thought bloat (causes latency failure in quant trading)
        (
            "Let's analyze this carefully. First, the SEC approval of Spot Ethereum ETF options "
            "represents an expansion of institutional derivative access. BlackRock and Fidelity "
            "will be able to offer structured yield and covered call strategies. Therefore, this "
            "is strongly bullish for ETH spot markets. Here is the output: "
            + json.dumps(ground_truth)
        ),

        # Candidate 5: Completely broken / unparseable JSON
        '{"ticker": "ETH", "catalyst_type": "REGULATORY_RULING", missing_quotes: true',
    ]

    labels = [
        "Candidate 0 (Perfect Raw JSON)",
        "Candidate 1 (Markdown-Wrapped JSON)",
        "Candidate 2 (Extra Hallucinated Key)",
        "Candidate 3 (Inverted Alpha Direction)",
        "Candidate 4 (Verbose CoT Bloat >350c)",
        "Candidate 5 (Malformed / Broken JSON)",
    ]

    # 2. Instantiate Vexel reward functions
    schema_reward_fn = pydantic_schema_compliance_reward(
        CatalystSignalSchema,
        strict_extra_penalty=0.35,
        missing_key_penalty=0.25,
    )
    concise_reward_fn = conciseness_latency_reward(
        max_tokens_target=48,
        char_penalty_threshold=220,
        decay_lambda=0.01,
    )

    # 3. Simulate trl.GRPOTrainer calling each reward function with (prompts, completions, **kwargs)
    kwargs = {"ground_truth": [ground_truth] * len(completions), "metadata": {"track": "catalyst"}}

    r_parse = strict_json_parse_reward(prompts, completions, **kwargs)
    r_schema = schema_reward_fn(prompts, completions, **kwargs)
    r_concise = concise_reward_fn(prompts, completions, **kwargs)
    r_alpha = directional_alpha_reward(prompts, completions, **kwargs)

    # Composite weights
    w_parse = 0.25
    w_schema = 0.35
    w_concise = 0.15
    w_alpha = 0.25

    total_rewards = []
    print(f"\n{'Candidate Description':<42} | {'Parse':<7} | {'Schema':<7} | {'Concise':<7} | {'Alpha':<7} | {'Total':<7}")
    print("-" * 88)
    for i in range(len(completions)):
        tot = (w_parse * r_parse[i]) + (w_schema * r_schema[i]) + (w_concise * r_concise[i]) + (w_alpha * r_alpha[i])
        total_rewards.append(tot)
        print(f"{labels[i]:<42} | {r_parse[i]:>7.2f} | {r_schema[i]:>7.2f} | {r_concise[i]:>7.2f} | {r_alpha[i]:>7.2f} | {tot:>7.3f}")

    # 4. Compute Group Relative Policy Optimization (GRPO) Normalized Advantages:
    # A_i = (R_i - mean(R)) / (std(R) + epsilon)
    mean_r = float(np.mean(total_rewards))
    std_r = float(np.std(total_rewards))
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in total_rewards]

    print("\n" + "=" * 80)
    print(" GRPO Group Relative Advantages (A_i = (R_i - mean) / (std + eps)):")
    print("=" * 80)
    for i, adv in enumerate(advantages):
        bar_len = int(max(0, (adv + 2.0) * 8))
        bar = "#" * bar_len
        print(f"[{i}] {labels[i]:<42} -> Advantage: {adv:>+6.3f} | {bar}")

    # 5. Assertions verifying mathematical correctness of reward gradients
    assert total_rewards[0] > total_rewards[1], "Raw JSON must score higher than markdown-wrapped JSON"
    assert total_rewards[0] > total_rewards[2], "Clean schema must score higher than hallucinated extra keys"
    assert total_rewards[0] > total_rewards[3], "Correct alpha direction must score higher than inverted direction"
    assert total_rewards[0] > total_rewards[4], "Concise output must score higher than verbose CoT bloat"
    assert total_rewards[0] > total_rewards[5], "Valid JSON must score higher than broken JSON"
    assert total_rewards[5] < 0.0, "Broken JSON must receive negative total reward penalty"
    assert advantages[0] > 0.0, "Perfect candidate must have positive relative advantage"
    assert advantages[5] < 0.0, "Broken candidate must have negative relative advantage"

    print("\n[SUCCESS] All reward gradients & GRPO relative advantage calculations verified 100%.")


if __name__ == "__main__":
    main()
