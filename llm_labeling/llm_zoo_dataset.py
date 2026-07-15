"""Convert optimized LLM zoo expressions into tensors and scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
import torch
from openai import OpenAI

from alphagen.config import CONSTANTS, DELTA_TIMES, MAX_EXPR_LENGTH, OPERATORS
from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    PairRollingOperator,
    RollingOperator,
    RollingConstOperator,
    UnaryOperator,
)
from alphagen.data.tokens import ConstantToken, DeltaTimeToken, FeatureToken, OperatorToken, SequenceIndicatorToken, SequenceIndicatorType, Token
from alphagen.rl.env.wrapper import (
    OFFSET_CONSTANT,
    OFFSET_DELTA_TIME,
    OFFSET_FEATURE,
    OFFSET_OP,
    OFFSET_SEP,
    SIZE_ACTION,
)
from alphagen_qlib.stock_data import FeatureType
from gan.network.masker import NetM
from gan.utils.factor_agent import ExpressionParser


LOGGER = logging.getLogger(__name__)



def _normalize_expression_string(expr: str) -> str:
    """Rewrite expressions to match available operator/token sets."""
    normalized = re.sub(r"Sqrt\s*\(", "Sqrt(", expr)
    if "Sqrt(" not in normalized:
        return normalized
    marker = "Sqrt("
    while True:
        start_idx = normalized.find(marker)
        if start_idx == -1:
            break
        inner_start = start_idx + len(marker)
        depth = 1
        cursor = inner_start
        length = len(normalized)
        while cursor < length and depth > 0:
            char = normalized[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1
        if depth != 0:
            # Unbalanced parentheses; abort further rewrites.
            break
        inner_expr = normalized[inner_start : cursor - 1]
        replacement = f"({inner_expr})**0.5"
        normalized = normalized[:start_idx] + replacement + normalized[cursor:]
    return normalized


# Expression -> token -> action utilities


def _expression_to_tokens(expr: Expression) -> List[Token]:
    tokens: List[Token] = []

    def visit(node: Expression) -> None:
        if isinstance(node, Feature):
            feature = node._feature   
            if not isinstance(feature, FeatureType):
                feature = FeatureType(int(feature))
            tokens.append(FeatureToken(feature))
        elif isinstance(node, Constant):
            value = float(node._value)   
            tokens.append(ConstantToken(value))
        elif isinstance(node, UnaryOperator):
            visit(node._operand)   
            tokens.append(OperatorToken(type(node)))
        elif isinstance(node, BinaryOperator):
            visit(node._lhs)   
            visit(node._rhs)   
            tokens.append(OperatorToken(type(node)))
        elif isinstance(node, RollingOperator):
            visit(node._operand)   
            tokens.append(DeltaTimeToken(int(node._delta_time)))   
            tokens.append(OperatorToken(type(node)))
        elif isinstance(node, RollingConstOperator):
            visit(node._operand)   
            tokens.append(DeltaTimeToken(int(node._delta_time)))   
            visit(node._const)   
            tokens.append(OperatorToken(type(node)))
        elif isinstance(node, PairRollingOperator):
            visit(node._lhs)   
            visit(node._rhs)   
            tokens.append(DeltaTimeToken(int(node._delta_time)))   
            tokens.append(OperatorToken(type(node)))
        else:
            raise TypeError(f"Unsupported expression node: {type(node)}")

    visit(expr)
    return tokens


def _token_to_action(token: Token) -> int:
    if isinstance(token, OperatorToken):
        try:
            op_idx = OPERATORS.index(token.operator)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Operator {token.operator} not in OPERATORS list") from exc
        return OFFSET_OP + op_idx - 1
    if isinstance(token, FeatureToken):
        feature_idx = int(token.feature)
        return OFFSET_FEATURE + feature_idx - 1
    if isinstance(token, DeltaTimeToken):
        try:
            dt_idx = DELTA_TIMES.index(int(token.delta_time))
        except ValueError as exc:
            raise ValueError(f"Delta time {token.delta_time} not supported") from exc
        return OFFSET_DELTA_TIME + dt_idx - 1
    if isinstance(token, ConstantToken):
        try:
            const_idx = CONSTANTS.index(float(token.constant))
        except ValueError as exc:
            raise ValueError(f"Constant {token.constant} not in CONSTANTS list") from exc
        return OFFSET_CONSTANT + const_idx - 1
    if isinstance(token, SequenceIndicatorToken):
        if token.indicator != SequenceIndicatorType.SEP:
            raise ValueError("Only SEP indicator is supported in action mapping")
        return OFFSET_SEP - 1
    raise TypeError(f"Unsupported token type: {type(token)}")


def expression_to_actions(expr: Expression, max_len: int) -> List[int]:
    tokens = _expression_to_tokens(expr)
    actions = [_token_to_action(token) for token in tokens]
    sep_action = _token_to_action(SequenceIndicatorToken(SequenceIndicatorType.SEP))
    actions.append(sep_action)
    if len(actions) > max_len:
        raise ValueError(f"Expression exceeds max length {max_len}: {expr}")
    actions.extend([sep_action] * (max_len - len(actions)))
    return actions


# Dataset construction


@dataclass
class DatasetBuildResult:
    expressions: List[str]
    logit_raw: torch.Tensor
    masked_logits: torch.Tensor
    onehot_tensor: torch.Tensor


def build_dataset(expressions: Sequence[str], max_len: int = MAX_EXPR_LENGTH) -> DatasetBuildResult:
    parser = ExpressionParser()
    parsed: List[Tuple[Expression, str]] = []
    failed: List[Tuple[str, Exception]] = []
    for expr_str in expressions:
        candidate = _normalize_expression_string(expr_str)
        try:
            parsed.append((parser.parse(candidate), candidate))
        except Exception as exc:  # noqa: BLE001
            failed.append((expr_str, exc))

    if failed:
        for expr_str, exc in failed:
            LOGGER.warning("Skipping expression %s: %s", expr_str, exc)

    valid_exprs = parsed
    if not valid_exprs:
        raise ValueError("No valid expressions available to build dataset")

    kept_exprs: List[Expression] = []
    action_sequences: List[List[int]] = []
    for expr, expr_text in valid_exprs:
        try:
            actions = expression_to_actions(expr, max_len)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to convert expression %s: %s", expr_text, exc)
            if isinstance(exc, ValueError):
                print(f"ExpressionTransformer fine-tune failed: {exc}")
            continue
        kept_exprs.append(expr)
        action_sequences.append(actions)

    if not action_sequences:
        raise ValueError("All expressions failed to convert into action sequences")

    logit_rows = []
    for actions in action_sequences:
        row = torch.full((max_len, SIZE_ACTION), -1e8, dtype=torch.float32)
        for step, action in enumerate(actions):
            row[step, action] = 10.0
        logit_rows.append(row)

    logit_raw = torch.stack(logit_rows, dim=0)
    netm = NetM(max_len=max_len, size_action=SIZE_ACTION)
    masked_logits, _, blds = netm(logit_raw)

    tokens_matrix = np.asarray(blds.builders_tokens, dtype=np.int64)
    expected = torch.as_tensor(action_sequences, dtype=torch.int64)
    recovered = torch.as_tensor(tokens_matrix, dtype=torch.int64)
    if not torch.equal(expected, recovered):
        raise ValueError("Recovered token sequence mismatch; expression may be invalid")
    onehot_tensor = torch.from_numpy(np.eye(SIZE_ACTION, dtype=np.float32)[tokens_matrix])

    return DatasetBuildResult(
        expressions=[str(expr) for expr in kept_exprs],
        logit_raw=logit_raw,
        masked_logits=masked_logits,
        onehot_tensor=onehot_tensor,
    )



# LLM scoring 


class LLMScoringError(RuntimeError):
    """Raised when the LLM response cannot be parsed reliably."""


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(seq), size):
        yield list(seq[start:start + size])


def _clean_response(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"(\[\s*[\s\S]*\s*\])", text)
    return match.group(1) if match else text


def score_with_llm(
    expressions: Sequence[str],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    batch_size: int = 5,
    temperature: float = 0.5,
) -> List[Dict[str, object]]:
    client_kwargs: Dict[str, object] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    results: List[Dict[str, object]] = []
    for batch in _chunked(list(expressions), max(1, batch_size)):
        prompt = json.dumps(batch, ensure_ascii=False)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(factor_list=prompt)}],
            temperature=temperature,
        )
        content = completion.choices[0].message.content or ""
        cleaned = _clean_response(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMScoringError(f"Failed to parse LLM output: {cleaned}") from exc
        if not isinstance(parsed, list) or len(parsed) != len(batch):
            raise LLMScoringError("LLM output size mismatch")
        for expr, item in zip(batch, parsed):
            if not isinstance(item, dict) or "score" not in item:
                raise LLMScoringError(f"Invalid LLM item: {item}")
            score_val = float(item.get("score", 0.0)) / 100.0
            score_val = max(0.0, min(1.0, score_val))
            score_val = round(score_val, 4)
            results.append({"expression": expr, "score": score_val})
    return results


PROMPT_TEMPLATE = (
    "You are a quantitative investment analyst. Please evaluate the quality of the following factor expressions.\n"
    "Scoring Criteria (Total 0-100):\n"
    "Financial Logic & Rationality (40 pts): Whether the factor is based on sound market logic or economic principles.\n"
    "Construction Quality & Complexity (30 pts): Ingenuity of expression design, avoidance of overfitting, meaningful feature combinations.\n"
    "Computational Robustness (20 pts): Data availability, handling of outliers, calculation stability.\n"
    "Expected Alpha Potential (10 pts): Degree to which the factor might provide unique predictive power for returns.\n"
    "Scoring Requirements:\n"
    "Use integer scores from 0 to 100. Do not limit scores to multiples of 5 or 10.\n"
    "Maximize score dispersion: Excellent factors should score close to 100 (90+). Flawed factors should score close to 0 (<20).\n"
    "Ensure clear distinction between factors within the same list. Avoid clustering scores.\n"
    "Bonus points may be given for complex expressions with sound underlying logic.\n"
    "Completely meaningless or obviously erroneous expressions can be scored 0-10.\n"
    "We also prefer longer factors, as this aligns with the goal of automated search.\n"
    "Factor list: {factor_list}\n"
    "Please return **a pure JSON array only**, without any Markdown code blocks.\n"
    "The array length should match the factor list, and each element should be an object containing:\n"
    "   - factor: the factor expression\n"
    "   - score: numeric score (0-100)"
)



DEFAULT_MODEL = "LLMs Name"
DEFAULT_API_KEY = "API Key"
DEFAULT_BASE_URL = "URL"


def load_llm_zoo(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"LLM zoo file not found: {path}")
    expressions: List[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            candidates = [
                reader.fieldnames.index(col)
                for col in ("normalized_expression", "expression")
                if col in reader.fieldnames
            ] if reader.fieldnames else []
            if not candidates:
                raise ValueError("LLM zoo CSV must contain 'normalized_expression' or 'expression'")
            key = "normalized_expression" if "normalized_expression" in reader.fieldnames else "expression"
            for row in reader:
                expr = (row.get(key) or "").strip()
                if expr:
                    expressions.append(expr)
    elif path.suffix.lower() == ".pkl":
        import pickle

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, list):
            for item in payload:
                expr = (item.get("normalized_expression") or item.get("expression") or "").strip()
                if expr:
                    expressions.append(expr)
        else:
            raise ValueError("Unsupported PKL structure for LLM zoo")
    else:
        raise ValueError("Unsupported LLM zoo format; please use CSV or PKL")

    if not expressions:
        raise ValueError("No expressions found in LLM zoo")
    return expressions


def write_scores_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_weight = any("weight" in row for row in rows)
    fieldnames = ["expression", "score"] + (["weight"] if has_weight else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "expression": row.get("expression"),
                "score": row.get("score"),
            }
            if has_weight:
                payload["weight"] = row.get("weight", 1.0)
            writer.writerow(payload)


def save_dataset_pt(path: Path, result: DatasetBuildResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expressions": result.expressions,
            "logit_raw": result.logit_raw,
            "masked_logits": result.masked_logits,
            "onehot_tensor": result.onehot_tensor,
        },
        path,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build tensors and LLM scores for LLM zoo expressions")
    parser.add_argument("--llm-zoo", type=Path, required=True, help="Path to llm_zoo CSV or PKL")
    parser.add_argument("--output-pt", type=Path, required=True, help="Where to store the tensor dataset")
    parser.add_argument("--output-scores", type=Path, required=True, help="Where to store LLM scores CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM API key")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM endpoint base URL")
    parser.add_argument("--batch-size", type=int, default=5, help="LLM request batch size")
    parser.add_argument("--temperature", type=float, default=0.5, help="LLM sampling temperature")
    parser.add_argument("--max-len", type=int, default=MAX_EXPR_LENGTH, help="Maximum expression length")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.api_key:
        raise SystemExit("API key must be provided via --api-key or environment variable")

    expressions = load_llm_zoo(args.llm_zoo)
    dataset = build_dataset(expressions, max_len=args.max_len)
    save_dataset_pt(args.output_pt, dataset)

    scores = score_with_llm(
        dataset.expressions,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    write_scores_csv(args.output_scores, scores)
    print(f"Processed {len(dataset.expressions)} expressions. Dataset saved to {args.output_pt}")
    print(f"LLM scores saved to {args.output_scores}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc: 
        print(f"Error: {exc}")
        raise
