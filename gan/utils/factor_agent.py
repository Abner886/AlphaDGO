import ast
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from alphagen.data.expression import (
    Abs,
    Add,
    Corr,
    Cov,
    Delta,
    Div,
    Ema,
    Expression,
    Inv,
    Ir,
    Kurt,
    Log,
    Sqrt,
    Mad,
    Max,
    Max_diff,
    Mean,
    Med,
    Min,
    Min_diff,
    Min_max_diff,
    Mul,
    Pctchange,
    Percentile,
    Pow,
    Regresi,
    Ref,
    S_log1p,
    Sequence,
    Sign,
    Skew,
    Std,
    Sub,
    Sum,
    Var,
    Wma,
    ts_div,
    ts_quantile,
    ts_Zscore,
)
from alphagen_generic.features import close, high, low, open_, target, volume, vwap

LOGGER = logging.getLogger(__name__)

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Mod,
    ast.Tuple,
)


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax: {ast.dump(node, include_attributes=False)}")
        if isinstance(node, ast.Call) and node.keywords:
            raise ValueError("Keyword arguments are not supported in factor expressions.")


def _build_eval_env() -> Dict[str, object]:
    env: Dict[str, object] = {
        "Abs": Abs,
        "Add": Add,
        "Corr": Corr,
        "Cov": Cov,
        "Delta": Delta,
        "Div": Div,
        "Ema": Ema,
        "Inv": Inv,
        "Ir": Ir,
        "Kurt": Kurt,
        "Log": Log,
        "Sqrt": Sqrt,
        "Mad": Mad,
        "Max": Max,
        "Max_diff": Max_diff,
        "Mean": Mean,
        "Med": Med,
        "Min": Min,
        "Min_diff": Min_diff,
        "Min_max_diff": Min_max_diff,
        "Mul": Mul,
        "Pctchange": Pctchange,
        "Percentile": Percentile,
        "Pow": Pow,
        "Regresi": Regresi,
        "Ref": Ref,
        "S_log1p": S_log1p,
        "Sequence": Sequence,
        "Sign": Sign,
        "Skew": Skew,
        "Std": Std,
        "Sub": Sub,
        "Sum": Sum,
        "Var": Var,
        "Wma": Wma,
        "ts_div": ts_div,
        "ts_quantile": ts_quantile,
        "PERCENTILE": Percentile,
        "REGRESI": Regresi,
        "SEQUENCE": Sequence,
        "ts_Zscore": ts_Zscore,
        "close": close,
        "high": high,
        "low": low,
        "open_": open_,
        "target": target,
        "volume": volume,
        "vwap": vwap,
        "open": open_,
    }
    return env


class ExpressionParser:
    def __init__(self) -> None:
        self._env = _build_eval_env()

    def parse(self, expr_str: str) -> Expression:
        tree = ast.parse(expr_str, mode="eval")
        _validate_ast(tree)
        compiled = compile(tree, "<factor>", "eval")
        expr = eval(compiled, {"__builtins__": {}}, self._env)
        if not isinstance(expr, Expression):
            raise ValueError(f"Expression did not produce a valid object: {expr_str}")
        if not expr.is_featured:
            raise ValueError(f"Expression missing feature input: {expr_str}")
        return expr

    def normalize(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return str(expr)

DEFAULT_PROMPT = (
    "You are a quantitative investment researcher tasked with factor optimization. "
    "Your objective is to generate {n} distinct, improved versions of a given factor expression. "
    "Each optimized version must enhance the factor's economic rationale and interpretability while preserving its core logical structure.\n\n"
    "**Available Tools:**\n"
    "You must construct your formulas using **only** the following elements:\n\n*   "
    "**Operators:** `+`, `-`, `*`, `/`, `**`\n"
    "**Functions:** `Abs(x)`, `Log(x)`, "
    "`Inv(x): reciprocal of x, i.e., 1/x`, "
    "`S_log1p(x): sign(x) x log(1 + |x|)`, "
    "`Sqrt(x): square root of x with negative inputs clipped to zero`, "
    "`Ref(x, d): value of x d days ago`, "
    "`Mean(x, d): mean of x over the past d days`, "
    "`Sum(x, d): sum of x over the past d days`, "
    "`Std(x, d): standard deviation of x over the past d days`, "
    "`Var(x, d): variance of x over the past d days`, "
    "`Skew(x, d): skewness of x over the past d days`, "
    "`Kurt(x, d): kurtosis of x over the past d days`, "
    "`Max(x, d): maximum of x over the past d days`, "
    "`Min(x, d): minimum of x over the past d days`, "
    "`Med(x, d): median of x over the past d days`, "
    "`Mad(x, d): mean absolute deviation of x over the past d days`, "
    "`ts_div(x, d): x divided by its value d days ago`, "
    "`Pctchange(x, d): percentage change of x compared to d days ago`, "
    "`Ir(x, d): mean(x, d) divided by std(x, d)`, "
    "`Delta(x, d): x minus its value d days ago`, "
    "`Wma(x, d): weighted moving average of x over the past d days`, "
    "`Ema(x, d): exponential moving average of x over the past d days`, "
    "`Cov(x, y, d): covariance of x and y over the past d days`, "
    "`Corr(x, y, d): correlation of x and y over the past d days`.\n"
    "**Windows (`d`):** Must be one of: `[1, 5, 10, 20, 30, 40, 50]`.\n"
    "**Constants:** Must be one of: `[-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]`.\n\n"
    "**Optimization Guidelines:**\n"
    "1.  **Enhance Rationality:** Make the factor's economic intuition clearer. For example, consider using ratios instead of raw differences, or apply transformations that better reflect investor behavior (e.g., `S_log1p` for signed, dampened growth).\n"
    "2.  **Improve Robustness:** Incorporate normalization (e.g., dividing by volatility or a rolling mean), smoothing (e.g., using `Wma`, `Ema`, `Mean`), or outlier handling (e.g., using `Med` or `S_log1p`)."
    "3.  **Maintain Framework:** Do not fundamentally alter the original factor's intent. If the original is a momentum signal, keep it as a momentum signal. Optimization should refine, not reinvent.\n"
    "4.  **Ensure Validity:** All formulas must be syntactically correct using only the allowed tools.\n\n"
    "**Task:**\n"
    "Original Factor: {factor}\n\n"
    "Generate exactly {n} optimized versions. Output **only** a single JSON object where the keys are `factor1`, `factor2`, ..., `factor{n}` and the values are the complete, optimized factor strings.\n\n"
    "-Output format example: {{\"factor1\": \"Abs(close)\", \"factor2\": \"Mean(close,5)\", ...}}\n"
)

@dataclass
class FactorOptimizationResult:
    original_factor: str
    raw_candidates: Dict[str, str]
    normalized_candidates: Dict[str, str]
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    round_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class FactorOptimizationAgent:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.deepseek.com",
        api_key: Optional[str] = None,
        parser: Optional[ExpressionParser] = None,
        prompt_template: str = DEFAULT_PROMPT,
        request_timeout: float = 60.0,
    ) -> None:
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("API key not provided. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
        self._client = OpenAI(api_key=resolved_key, base_url=base_url, timeout=request_timeout)
        self._parser = parser or ExpressionParser()
        self._prompt_template = prompt_template
        self._model = model

    def build_prompt(self, factor: str, n: int) -> str:
        return self._prompt_template.format(factor=factor, n=n)

    def parse_expression(self, expr_text: str) -> Expression:
        return self._parser.parse(expr_text)

    def optimize(self, factor: str, n: int) -> FactorOptimizationResult:
        prompt = self.build_prompt(factor, n)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        message = response.choices[0].message.content or ""
        print(message)
        raw_dict = self._parse_json(message)
        normalized: Dict[str, str] = {}
        for key, expr_text in raw_dict.items():
            try:
                normalized[key] = self._parser.normalize(expr_text)
            except Exception as exc:
                LOGGER.warning("Failed to normalize %s: %s", key, exc)
        return FactorOptimizationResult(
            original_factor=factor,
            raw_candidates=raw_dict,
            normalized_candidates=normalized,
        )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            if stripped.endswith("```"):
                stripped = stripped.rsplit("```", 1)[0]
        return stripped.strip()

    def _parse_json(self, text: str) -> Dict[str, str]:
        stripped = self._strip_markdown(text)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {stripped}") from exc
        if not isinstance(data, dict):
            raise ValueError("Model output must be a JSON object with factor keys.")
        result: Dict[str, str] = {}
        for key, value in data.items():
            if not isinstance(value, str):
                LOGGER.warning("Skipping non-string entry %s: %r", key, value)
                continue
            result[str(key)] = value.strip()
        return result


class FactorOptimizationWriter:
    def __init__(self, output_dir: Path, base_filename: str = "optimized_factors") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._base_filename = base_filename

    def save_batch(self, results: Iterable[FactorOptimizationResult]) -> Optional[Tuple[Path, Path]]:
        records: List[Dict[str, object]] = []
        for item in results:
            for key, expr in item.normalized_candidates.items():
                records.append(
                    {
                        "round_index": item.round_index,
                        "original_factor": item.original_factor,
                        "candidate_key": key,
                        "normalized_expression": expr,
                        "raw_expression": item.raw_candidates.get(key, expr),
                        "score": item.candidate_scores.get(key) if item.candidate_scores else None,
                        "timestamp": item.timestamp,
                    }
                )
        if not records:
            return None
        df = pd.DataFrame(records)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self._output_dir / f"{self._base_filename}_{stamp}.csv"
        pkl_path = self._output_dir / f"{self._base_filename}_{stamp}.pkl"
        df.to_csv(csv_path, index=False)
        with open(pkl_path, "wb") as handle:
            pickle.dump(records, handle)
        return csv_path, pkl_path
