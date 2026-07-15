from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import math
import torch

from alphagen.utils.pytorch_utils import normalize_by_day

ScoreFn = Callable[[torch.Tensor, torch.Tensor], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]


@dataclass
class NoiseRobustnessConfig:
    """Configuration for stochastic noise robustness evaluation."""

    noise_scale: float = 0.15
    num_samples: int = 3
    penalty_weight: float = 0.5
    tolerance: float = 0.0  # acceptable drop ratio before penalising
    metric_target: str = "ic"
    renormalize: bool = True
    min_std: float = 1e-4
    time_stability_target: float = 0.7
    time_stability_weight: float = 0.0
    # linear | geometric | harmonic | product | softmin
    blend_mode: str = "linear"
    market_exponent: float = 1.0
    time_exponent: float = 1.0
    softmin_temperature: float = 1.0


class NoiseRobustnessEvaluator:

    def __init__(self, config: NoiseRobustnessConfig):
        self.config = config

    def _perturb(self, factor: torch.Tensor) -> torch.Tensor:
        std = factor.std(dim=1, keepdim=True)
        std = std.clamp_min(self.config.min_std)
        noise = torch.randn_like(factor) * std * self.config.noise_scale
        perturbed = factor + noise
        if self.config.renormalize:
            perturbed = normalize_by_day(perturbed)
        return perturbed

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        ratio = numerator / denominator
        if not math.isfinite(ratio):
            return 0.0
        return max(0.0, min(1.0, ratio))

    def _time_ratio(
        self,
        base_time_stability: Optional[float],
        noise_time_stabilities: Optional[list],
    ) -> float:
        target_ts = max(self.config.time_stability_target, 1e-8)

        base_ratio = 0.0
        if base_time_stability is not None and math.isfinite(base_time_stability):
            base_ratio = max(0.0, min(1.0, base_time_stability / target_ts))

        stability_ratio = base_ratio
        if noise_time_stabilities:
            min_noise_ts = min(noise_time_stabilities)
            noise_ratio = max(0.0, min(1.0, min_noise_ts / target_ts))
            stability_ratio = min(stability_ratio, noise_ratio)

        return stability_ratio

    def _temporal_stability(self, factor: torch.Tensor) -> Optional[float]:
        if factor.dim() != 2:
            return None

        mask = torch.isfinite(factor)
        valid_rows = mask.any(dim=1)
        if not bool(valid_rows.any()):
            return None

        factor = factor[valid_rows]
        mask = mask[valid_rows]
        if factor.size(0) < 2:
            return None

        ranked = factor.argsort(dim=1).argsort(dim=1).float()
        ranked[~mask] = 0.0

        row_sum = ranked.sum(dim=1, keepdim=True)
        valid_rows = (row_sum.squeeze(1) > 0)
        if not bool(valid_rows.any()):
            return None

        ranked = ranked[valid_rows]
        mask = mask[valid_rows]
        row_sum = row_sum[valid_rows]
        if ranked.size(0) < 2:
            return None

        probs = ranked / row_sum
        probs[~mask] = 0.0

        curr = probs[1:]
        prev = probs[:-1]
        curr_mask = mask[1:]
        prev_mask = mask[:-1]
        if curr.size(0) == 0:
            return None

        pair_mask = curr_mask & prev_mask
        if not bool(pair_mask.any()):
            return None

        curr = curr * pair_mask
        prev = prev * pair_mask

        curr_sum = curr.sum(dim=1, keepdim=True)
        prev_sum = prev.sum(dim=1, keepdim=True)
        valid_pairs = (curr_sum.squeeze(1) > 0) & (prev_sum.squeeze(1) > 0)
        if not bool(valid_pairs.any()):
            return None

        curr = curr[valid_pairs]
        prev = prev[valid_pairs]
        curr_sum = curr_sum[valid_pairs]
        prev_sum = prev_sum[valid_pairs]

        curr = curr / curr_sum
        prev = prev / prev_sum

        eps = 1e-8
        ratio = torch.log((curr + eps) / (prev + eps))
        kl = (curr * ratio).sum(dim=1)
        finite_mask = torch.isfinite(kl)
        if not bool(finite_mask.any()):
            return None

        kl = kl[finite_mask].clamp_min(0.0)
        if kl.numel() == 0:
            return None

        rre = 1.0 / (1.0 + kl)
        finite_rre = torch.isfinite(rre)
        if not bool(finite_rre.any()):
            return None

        return float(rre[finite_rre].mean().item())

    def __call__(
        self,
        factor: torch.Tensor,
        target: torch.Tensor,
        base_score: float,
        score_fn: ScoreFn,
        _base_multi_score: Optional[Dict[str, float]] = None,
    ) -> float:
        base_time_stability = self._temporal_stability(factor)
        if _base_multi_score is not None:
            if base_time_stability is None or not math.isfinite(base_time_stability):
                _base_multi_score["time_stability"] = 0.0
            else:
                _base_multi_score["time_stability"] = float(base_time_stability)

        noise_scores = []
        noise_time_stabilities = []
        if self.config.num_samples > 0 and self.config.noise_scale > 0:
            for _ in range(self.config.num_samples):
                noisy_factor = self._perturb(factor)
                multi_score_noisy, _, _ = score_fn(noisy_factor, target)
                noise_scores.append(float(multi_score_noisy.get(self.config.metric_target, 0.0)))
                ts = self._temporal_stability(noisy_factor)
                if ts is not None and math.isfinite(ts):
                    noise_time_stabilities.append(float(ts))

        base_score_float = float(base_score)
        adjusted_market = base_score_float
        drop_ratio = 0.0
        if noise_scores:
            min_noise_score = float(min(noise_scores))
            drop = max(0.0, base_score_float - min_noise_score)
            tolerated_drop = max(0.0, base_score_float) * max(0.0, self.config.tolerance)
            penalty = max(0.0, drop - tolerated_drop)
            adjusted_market = max(0.0, base_score_float - self.config.penalty_weight * penalty)
            if base_score_float > 0.0:
                drop_ratio = drop / base_score_float

        market_ratio = self._safe_ratio(adjusted_market, base_score_float)

        if _base_multi_score is not None:
            _base_multi_score["market_stability"] = float(market_ratio)
            if base_score_float > 0.0:
                _base_multi_score["noise_drop_ratio"] = float(drop_ratio)
                _base_multi_score["tolerated_drop_ratio"] = max(0.0, self.config.tolerance)

        time_ratio = self._time_ratio(base_time_stability, noise_time_stabilities)

        if _base_multi_score is not None:
            _base_multi_score["time_stability_ratio"] = float(time_ratio)
            if noise_time_stabilities:
                _base_multi_score["time_stability_noisy_min"] = float(min(noise_time_stabilities))

        time_weight = max(0.0, min(1.0, self.config.time_stability_weight))
        market_weight = 1.0 - time_weight

        mode = (self.config.blend_mode or "linear").lower()
        me = max(0.0, self.config.market_exponent)
        te = max(0.0, self.config.time_exponent)
        if mode == "linear":
            final_ratio = market_weight * market_ratio + time_weight * time_ratio
        elif mode == "geometric":
            eps = 1e-8
            final_ratio = (market_ratio + eps) ** (market_weight * me) * (time_ratio + eps) ** (time_weight * te)
            total_pow = market_weight * me + time_weight * te
            if total_pow > 0:
                final_ratio = final_ratio ** (1.0 / total_pow)
        elif mode == "harmonic":
            eps = 1e-8
            denom = market_weight / max(market_ratio, eps) + time_weight / max(time_ratio, eps)
            final_ratio = 0.0 if denom <= 0 else 1.0 / denom
        elif mode == "product":
            eps = 1e-8
            final_ratio = (market_ratio + eps) ** me * (time_ratio + eps) ** te
            total_pow = me + te
            if total_pow > 0:
                final_ratio = final_ratio ** (1.0 / total_pow)
        elif mode == "softmin":
            T = max(1e-6, self.config.softmin_temperature)
            w_m = math.exp(-market_ratio / T)
            w_t = math.exp(-time_ratio / T)
            final_ratio = (market_ratio * w_m + time_ratio * w_t) / (w_m + w_t)
        else:
            final_ratio = market_weight * market_ratio + time_weight * time_ratio

        final_ratio = max(0.0, min(1.0, float(final_ratio)))

        final_score = float(base_score) * final_ratio
        if _base_multi_score is not None:
            _base_multi_score["final_ratio"] = final_ratio
            _base_multi_score["blend_mode"] = mode
            _base_multi_score["market_exponent"] = me
            _base_multi_score["time_exponent"] = te
        return final_score
