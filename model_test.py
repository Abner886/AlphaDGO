import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from alphagen_generic.features import target
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from gan.utils.data import get_data_by_year



def evaluate_combinations(
    instruments: str = "csi300",
    freq: str = "day",
    save_name: str = "test",
    window_label: str = "inf",
    train_start: int = 2012,
    train_end_years: Sequence[int] = (2022,),
    seeds: Sequence[int] = (0,1,2,3,4,),
    n_factors_list: Sequence[int] = (10,),
    holding_period: int = 20,
    top_pct: float = 0.1,
    model_tag: str = "100",
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> pd.DataFrame:
    """Evaluate predictions produced by Alphs_combine."""
    results: List[Dict[str, float]] = []
    for n_factors in n_factors_list:
        for seed in seeds:
            ic_values: List[torch.Tensor] = []
            ric_values: List[torch.Tensor] = []
            ar_vals: List[float] = []
            ir_vals: List[float] = []
            for train_end in train_end_years:
                data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = get_data_by_year(
                    train_start=train_start,
                    train_end=train_end,
                    valid_year=train_end + 1,
                    test_year=train_end + 2,
                    instruments=instruments,
                    target=target,
                    freq=freq,
                )
                run_name = f"llm_{train_end}_{n_factors}_{window_label}_{seed}"
                output_dir = Path("out") / model_tag / f"{save_name}_{instruments}_{train_end}_{seed}"
                pred_path = output_dir / f"pred_{run_name}.pt"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Prediction file missing: {pred_path}")
                pred = torch.load(pred_path).to(device)
                tgt = target.evaluate(data_all).to(device)
                tgt = tgt[-data_test.n_days:]

                ic_tensor = torch.nan_to_num(batch_pearsonr(pred, tgt), nan=0.0)
                ric_tensor = torch.nan_to_num(batch_spearmanr(pred, tgt), nan=0.0)
                ic_values.append(ic_tensor)
                ric_values.append(ric_tensor)

            ic_cat = torch.cat(ic_values)
            ric_cat = torch.cat(ric_values)
            ic_mean = ic_cat.mean().item()
            ric_mean = ric_cat.mean().item()
            ic_std = ic_cat.std().item()
            ric_std = ric_cat.std().item()

            results.append(
                {
                    "n_factors": n_factors,
                    "seed": seed,
                    "ic_mean": ic_mean,
                    "ric_mean": ric_mean,
                    "icir": ic_mean / ic_std if ic_std > 1e-12 else float("nan"),
                    "ricir": ric_mean / ric_std if ric_std > 1e-12 else float("nan"),
                }
            )
    return pd.DataFrame(results)


def main(config_path: Path | None = None) -> None:
    """Run evaluation using optional JSON configuration."""
    kwargs: Dict[str, object]
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            kwargs = json.load(handle)
    else:
        kwargs = {}

    df = evaluate_combinations(**kwargs)
    grouped = df.groupby("n_factors").agg({
        "ic_mean": ["mean", "std"],
        "ric_mean": ["mean", "std"],
        "icir": ["mean", "std"],
        "ricir": ["mean", "std"],
    })

    # Ensure the full DataFrame output is visible in the console without truncation.
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        0,
    ):
        print(df.to_string(index=False))
        print(grouped.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM-combined factors.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path.")
    args = parser.parse_args()
    main(args.config)
