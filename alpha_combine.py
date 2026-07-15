import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alphagen_generic.features import target
from gan.utils.builder import exprs2tensor
from gan.utils.data import get_data_by_year
from gan.utils.factor_agent import ExpressionParser
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret


def load_llm_records(base_dir: Union[str, Path], filename: str = "llm_zoo.pkl") -> List[Dict[str, object]]:
    base_path = Path(base_dir)
    candidates: List[Path] = []
    if filename:
        candidates.append(base_path / filename)
    stem = Path(filename).stem if filename else "llm_zoo"
    if not candidates or not candidates[0].is_file():
        candidates.append(base_path / f"{stem}.pkl")
        candidates.append(base_path / f"{stem}.csv")
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".pkl":
            with open(path, "rb") as handle:
                records = pickle.load(handle)
            if isinstance(records, list):
                return records
            if isinstance(records, dict):
                nested = records.get("records")
                if isinstance(nested, list):
                    return nested
            raise ValueError(f"Unsupported pickle structure in {path}")
        if path.suffix == ".csv":
            return pd.read_csv(path).to_dict("records")
    available = sorted(base_path.glob("llm_zoo*"))
    raise FileNotFoundError(f"LLM record file not found in {base_path}. Checked {candidates}. Available: {available}")


def build_expression_frame(
    records: Sequence[Dict[str, object]],
    parser: Optional[ExpressionParser] = None,
    score_floor: float = 0.0,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    parser = parser or ExpressionParser()
    sorted_records = sorted(records, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    rows: List[Dict[str, object]] = []
    seen: set[str] = set()
    for record in sorted_records:
        expr_text = record.get("normalized_expression")
        if not expr_text:
            continue
        score = float(record.get("score", 0.0) or 0.0)
        if score < score_floor:
            continue
        if expr_text in seen:
            continue
        try:
            expr = parser.parse(expr_text)
        except Exception as exc:
            print(f"Skip expression '{expr_text}': {exc}")
            continue
        rows.append(
            {
                "exprs": expr,
                "expr_text": str(expr),
                "score": score,
                "original_factor": record.get("original_factor"),
            }
        )
        seen.add(expr_text)
        if max_items is not None and len(rows) >= max_items:
            break
    if not rows:
        return pd.DataFrame(columns=["exprs", "expr_text", "score", "original_factor"])
    return pd.DataFrame(rows)


def chunk_batch_spearmanr(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
    n_days = len(x)
    parts: List[torch.Tensor] = []
    for start in range(0, n_days, chunk_size):
        parts.append(batch_spearmanr(x[start:start + chunk_size], y[start:start + chunk_size]))
    return torch.cat(parts, dim=0)


def get_tensor_metrics_raw(x: torch.Tensor, y: torch.Tensor) -> Sequence[torch.Tensor]:
    ic_vals = batch_pearsonr(x, y)
    ric_vals = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_vals = batch_ret(x, y)
    ic_vals = torch.nan_to_num(ic_vals, nan=0.0)
    ric_vals = torch.nan_to_num(ric_vals, nan=0.0)
    ret_vals = torch.nan_to_num(ret_vals, nan=0.0)
    return ic_vals, ric_vals, ret_vals


def main(
    instruments: str = "csi300",
    train_end_year: int = 2022,
    freq: str = "day",
    seeds: str = "[0,1,2,3,4]",
    cuda: int = 0,
    save_name: str = "test",
    n_factors: int = 10,
    window: Union[int, str] = "inf",
    train_start: int = 2012,
    llm_dir: str = "llm",
    llm_filename: str = "llm_zoo.pkl",
    score_floor: float = 0.0,
    max_items: Optional[int] = None,
    model_tag: str = "100",
) -> None:
    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)
    if isinstance(window, str):
        assert window == "inf"
        window = float("inf")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    train_end = train_end_year
    data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = get_data_by_year(
        train_start=train_start,
        train_end=train_end,
        valid_year=train_end + 1,
        test_year=train_end + 2,
        instruments=instruments,
        target=target,
        freq=freq,
    )
    parser = ExpressionParser()
    for seed in seeds:
        base_dir = Path("out") / model_tag / f"{save_name}_{instruments}_{train_end}_{seed}"
        llm_path = base_dir / llm_dir
        try:
            records = load_llm_records(llm_path, llm_filename)
        except FileNotFoundError as exc:
            print(f"Skip seed {seed}: {exc}")
            continue
        df = build_expression_frame(records, parser, score_floor=score_floor, max_items=max_items)
        if df.empty:
            print(f"Skip seed {seed}: no valid LLM expressions after filtering.")
            continue
        expr_list = df["exprs"].tolist()
        fct_tensor = exprs2tensor(expr_list, data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)
        ic_list: List[torch.Tensor] = []
        ric_list: List[torch.Tensor] = []
        ret_list: List[torch.Tensor] = []
        for cur in tqdm(range(fct_tensor.shape[-1])):
            ic_vals, ric_vals, ret_vals = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
            ic_list.append(ic_vals)
            ric_list.append(ric_vals)
            ret_list.append(ret_vals)
        ic_vals = torch.stack(ic_list, dim=-1)
        ric_vals = torch.stack(ric_list, dim=-1)
        ret_vals = torch.stack(ret_list, dim=-1)
        torch.cuda.empty_cache()
        shift = 20
        pred_list: List[torch.Tensor] = []
        ics_list: List[np.ndarray] = []
        rics_list: List[np.ndarray] = []
        good_idx_list: List[List[int]] = []
        weight_records: List[Dict[str, object]] = []
        start_idx = len(fct_tensor) - data_test.n_days - data_valid.n_days
        window_label = "inf" if not np.isfinite(window) else int(window)
        run_name = f"llm_{train_end}_{n_factors}_{window_label}_{seed}"
        tensor_save_path = base_dir
        tensor_save_path.mkdir(parents=True, exist_ok=True)
        for cur in tqdm(range(start_idx, len(fct_tensor))):
            begin = 0 if not np.isfinite(window) else cur - int(window) - shift
            cur_ic = ic_vals[begin:cur - shift]
            cur_ric = ric_vals[begin:cur - shift]
            cur_ret = ret_vals[begin:cur - shift]
            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)
            ret_mean = cur_ret.mean(dim=0)
            ret_std = cur_ret.std(dim=0)
            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            retir = ret_mean / ret_std
            metrics = dict(
                ic=ic_mean.detach().cpu().numpy(),
                ic_std=ic_std.detach().cpu().numpy(),
                icir=icir.detach().cpu().numpy(),
                ric=ric_mean.detach().cpu().numpy(),
                ric_std=ric_std.detach().cpu().numpy(),
                ricir=ricir.detach().cpu().numpy(),
                ret=ret_mean.detach().cpu().numpy(),
                ret_std=ret_std.detach().cpu().numpy(),
                retir=retir.detach().cpu().numpy(),
            )
            metric_df = pd.DataFrame(metrics).sort_values("ricir", ascending=False, key=lambda x: abs(x))
            filtered = metric_df[(metric_df["ric"] > 0.02) & (metric_df["ricir"] > 0.2)]
            if len(filtered) < 1:
                filtered = metric_df.iloc[:1]
            good_idx = filtered.iloc[:n_factors].index.to_list()
            good_idx_list.append(good_idx)
            x = fct_tensor[begin:cur - shift, :, good_idx]
            y = tgt_tensor[begin:cur - shift]
            to_pred = fct_tensor[cur, :, good_idx]
            y_true = tgt_tensor[cur]
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])
            mask = torch.isfinite(y)[:, 0]
            y = y[mask]
            x = x[mask]
            to_pred = torch.nan_to_num(to_pred, nan=0.0)
            ones_train = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones_train], dim=-1)
            ones_pred = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones_pred], dim=-1)
            if x.numel() == 0:
                coef = torch.zeros((x.shape[1], y.shape[1]), device=x.device)
            else:
                try:
                    coef = torch.linalg.lstsq(x, y).solution
                except RuntimeError as exc:
                    print(f"Linear solve failed at step {cur}: {exc}")
                    coef = torch.zeros((x.shape[1], y.shape[1]), device=x.device)
            pred = to_pred @ coef
            weight_records.append(
                {
                    "step": int(cur),
                    "indices": good_idx,
                    "weights": coef.detach().cpu().numpy().tolist(),
                }
            )
            cur_ic_val = batch_pearsonr(pred.T, y_true.T)[0]
            cur_ric_val = batch_spearmanr(pred.T, y_true.T)[0]
            ics_list.append(cur_ic_val.detach().cpu().numpy())
            rics_list.append(cur_ric_val.detach().cpu().numpy())
            pred_list.append(pred[:, 0])
            torch.cuda.empty_cache()
        valid_len = data_valid.n_days
        test_len = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        torch.save(all_pred[-test_len - valid_len:-valid_len].detach().cpu(), tensor_save_path / f"pred_valid_{run_name}.pt")
        torch.save(all_pred[-test_len:].detach().cpu(), tensor_save_path / f"pred_{run_name}.pt")
        with open(tensor_save_path / f"weights_{run_name}.pkl", "wb") as handle:
            pickle.dump(weight_records, handle)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
