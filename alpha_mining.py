# Shi H, Song W, Zhang X, et al. AlphaForge (2025) [https://github.com/dulyhao/alphaforge]


import torch 
import os
import time
import pickle
from pathlib import Path
from collections import defaultdict

from typing import Dict, List, Optional, Sequence as TypingSequence, Set, Tuple
import pandas as pd
from sympy import factor_list

from gan.dataset import Collector
from gan.network.masker import NetM
from gan.network.predictor import NetPMamba, train_regression_model_with_weight
from gan.network.generater import NetG_Mamba
from alphagen.rl.env.wrapper import SIZE_ACTION
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.data.expression import BinaryOperator, Constant, Expression, Feature, PairRollingOperator, RollingOperator, RollingConstOperator, UnaryOperator
from alphagen.utils.correlation import batch_ret, batch_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
import numpy as np
from alphagen.utils.random import reseed_everything
from gan.utils import (
    NoiseRobustnessConfig,
    NoiseRobustnessEvaluator,
    filter_valid_blds,
    save_blds,
)
from gan.network.generater import ExpressionScorePredictor, train_network_generator
import gc
from gan.utils.data import get_data_by_year
from gan.utils.factor_agent import FactorOptimizationAgent, ExpressionParser
from alphagen.config import DELTA_TIMES as SUPPORTED_DELTA_TIMES, CONSTANTS as SUPPORTED_CONSTANTS
from llm_labeling.llm_zoo_dataset import build_dataset, save_dataset_pt, score_with_llm, write_scores_csv, expression_to_actions
from llm_labeling.train_expression_regressor import TrainingConfig as ExpressionTrainingConfig, train as run_expression_transformer_training


def _expression_action_length(expr: Expression) -> int:
    token_count = 0

    def visit(node: Expression) -> None:
        nonlocal token_count
        if isinstance(node, Feature) or isinstance(node, Constant):
            token_count += 1
        elif isinstance(node, UnaryOperator):
            visit(node._operand)  # type: ignore[attr-defined]
            token_count += 1
        elif isinstance(node, BinaryOperator):
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]
            token_count += 1
        elif isinstance(node, RollingOperator):
            visit(node._operand)  # type: ignore[attr-defined]
            token_count += 2  # delta time + operator
        elif isinstance(node, RollingConstOperator):
            visit(node._operand)  # type: ignore[attr-defined]
            visit(node._const)  # type: ignore[attr-defined]
            token_count += 2  # delta time + operator
        elif isinstance(node, PairRollingOperator):
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]
            token_count += 2  # delta time + operator
        else:
            raise TypeError(f"Unsupported expression node: {type(node)}")

    visit(expr)
    return token_count + 1  # include terminating SEP token


def _find_invalid_delta_times(expr: Expression, allowed: TypingSequence[int]) -> List[float]:
    allowed_set = {int(item) for item in allowed}
    invalid: List[float] = []

    def visit(node: Expression) -> None:
        if isinstance(node, RollingOperator):
            dt_value = float(getattr(node, "_delta_time", 0))
            int_dt = int(round(dt_value))
            if abs(dt_value - int_dt) > 1e-6 or int_dt not in allowed_set:
                invalid.append(dt_value)
            visit(node._operand)  # type: ignore[attr-defined]
        elif isinstance(node, RollingConstOperator):
            dt_value = float(getattr(node, "_delta_time", 0))
            int_dt = int(round(dt_value))
            if abs(dt_value - int_dt) > 1e-6 or int_dt not in allowed_set:
                invalid.append(dt_value)
            visit(node._operand)  # type: ignore[attr-defined]
            visit(node._const)  # type: ignore[attr-defined]
        elif isinstance(node, PairRollingOperator):
            dt_value = float(getattr(node, "_delta_time", 0))
            int_dt = int(round(dt_value))
            if abs(dt_value - int_dt) > 1e-6 or int_dt not in allowed_set:
                invalid.append(dt_value)
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]
        elif isinstance(node, UnaryOperator):
            visit(node._operand)  # type: ignore[attr-defined]
        elif isinstance(node, BinaryOperator):
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]

    visit(expr)
    unique_invalid = []
    for value in invalid:
        if value not in unique_invalid:
            unique_invalid.append(value)
    return unique_invalid


def _find_invalid_constants(expr: Expression, allowed: TypingSequence[float], tol: float = 1e-6) -> List[float]:
    allowed_vals = [float(item) for item in allowed]
    invalid: List[float] = []

    def is_allowed(val: float) -> bool:
        for ref in allowed_vals:
            if abs(val - ref) <= tol * max(1.0, abs(ref)):
                return True
        return False

    def visit(node: Expression) -> None:
        if isinstance(node, Constant):
            value = float(getattr(node, "_value", 0.0))
            if not is_allowed(value):
                invalid.append(value)
        elif isinstance(node, UnaryOperator):
            visit(node._operand)  # type: ignore[attr-defined]
        elif isinstance(node, BinaryOperator):
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]
        elif isinstance(node, RollingOperator):
            visit(node._operand)  # type: ignore[attr-defined]
        elif isinstance(node, RollingConstOperator):
            visit(node._operand)  # type: ignore[attr-defined]
            visit(node._const)  # type: ignore[attr-defined]
        elif isinstance(node, PairRollingOperator):
            visit(node._lhs)  # type: ignore[attr-defined]
            visit(node._rhs)  # type: ignore[attr-defined]

    visit(expr)
    unique_invalid: List[float] = []
    for value in invalid:
        if value not in unique_invalid:
            unique_invalid.append(value)
    return unique_invalid

def pre_process_y(y):
    min_y = 0
    max_y = y.flatten().max()
    y = (y - min_y) / (max_y - min_y) * 100
    return y
def numpy2onehot(integer_matrix,max_num_categories=None,min_num_categories=None):
    if max_num_categories is None:
        max_num_categories = np.max(integer_matrix) + 1
    if min_num_categories is None:
        min_num_categories = np.min(integer_matrix)
    integer_matrix = integer_matrix - min_num_categories
    num_categories = max_num_categories - min_num_categories
    return np.eye(num_categories)[integer_matrix]



def blds_list_to_tensor(blds_list,weights_list:List[int]):
    assert len(blds_list) == len(weights_list)

    x_numpy_list = []
    y_numpy_list = []
    weights_numpy_list = []
    for blds,weight_int in zip(blds_list,weights_list):
        x_numpy = numpy2onehot(np.array(blds.builders_tokens),SIZE_ACTION,0).astype('float32')
        y_numpy = np.array(blds.scores).astype('float32')[:,None]
        weights_numpy = np.ones(x_numpy.shape[0]).astype('float32')[:,None] * weight_int
        x_numpy_list.append(x_numpy)
        y_numpy_list.append(y_numpy)
        weights_numpy_list.append(weights_numpy)
    x_numpy = np.concatenate(x_numpy_list,axis=0)
    y_numpy = np.concatenate(y_numpy_list,axis=0)
    weights_numpy = np.concatenate(weights_numpy_list,axis=0)
    x = torch.from_numpy(x_numpy)
    y = torch.from_numpy(y_numpy)
    weights = torch.from_numpy(weights_numpy)
    return x,y,weights


def save_llm_factors(records: List[Dict[str, object]], output_dir: Path) -> None:
    if not records:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda x: x.get('score', 0.0), reverse=True)
    df = pd.DataFrame(sorted_records)
    df = df[['original_factor', 'normalized_expression', 'score']]
    csv_path = output_dir / "llm_zoo.csv"
    pkl_path = output_dir / "llm_zoo.pkl"
    df.to_csv(csv_path, index=False)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(sorted_records, handle)


def finetune_expression_transformer(
    cfg,
    expressions: TypingSequence[str],
    weights: Optional[TypingSequence[float]] = None,
) -> Tuple[Optional[ExpressionScorePredictor], List[str]]:
    if weights is not None and len(weights) != len(expressions):
        raise ValueError("Expression and weight counts must match for fine-tuning.")

    entries: List[Tuple[str, float]] = []
    skipped_expressions: Set[str] = set()
    if weights is None:
        for expr in expressions:
            clean = (expr or "").strip()
            if clean:
                entries.append((clean, 1.0))
    else:
        for expr, weight in zip(expressions, weights):
            clean = (expr or "").strip()
            if not clean:
                skipped_expressions.add(expr or "")
                continue
            value = float(weight)
            if value <= 0.0:
                skipped_expressions.add(clean)
                continue
            entries.append((clean, value))

    if not entries:
        return None, list(skipped_expressions)
    if not getattr(cfg, 'llm_api_key', None):
        raise RuntimeError("LLM API key required for ExpressionTransformer fine-tuning.")

    parser = ExpressionParser()
    raw_weight_map: Dict[str, float] = defaultdict(float)
    for expr, weight in entries:
        raw_weight_map[expr] += weight

    filtered_raw_weights: Dict[str, float] = {}
    normalized_weight_map: Dict[str, float] = defaultdict(float)
    for expr, weight in raw_weight_map.items():
        try:
            parsed_expr = parser.parse(expr)
        except Exception as parse_exc:
            print(f"ExpressionTransformer fine-tune skipped '{expr}': {parse_exc}")
            skipped_expressions.add(expr)
            continue
        normalized_text = str(parsed_expr)
        skipped_expressions.discard(normalized_text)
        try:
            action_len = _expression_action_length(parsed_expr)
        except Exception as len_exc:
            print(f"ExpressionTransformer fine-tune skipped '{expr}': {len_exc}")
            skipped_expressions.update({expr, normalized_text})
            continue
        if action_len > cfg.max_len:
            print(
                f"ExpressionTransformer fine-tune skipped '{expr}': length {action_len} exceeds max {cfg.max_len}."
            )
            skipped_expressions.update({expr, normalized_text})
            continue
        try:
            expression_to_actions(parsed_expr, cfg.max_len)
        except Exception as token_exc:
            print(
                f"ExpressionTransformer fine-tune skipped '{expr}': token conversion failed - {token_exc}"
            )
            skipped_expressions.update({expr, normalized_text})
            continue
        invalid_windows = _find_invalid_delta_times(parsed_expr, SUPPORTED_DELTA_TIMES)
        if invalid_windows:
            print(
                f"ExpressionTransformer fine-tune skipped '{expr}': unsupported window sizes {invalid_windows}."
            )
            skipped_expressions.update({expr, normalized_text})
            continue
        invalid_constants = _find_invalid_constants(parsed_expr, SUPPORTED_CONSTANTS)
        if invalid_constants:
            print(
                f"ExpressionTransformer fine-tune skipped '{expr}': unsupported constants {invalid_constants}."
            )
            skipped_expressions.update({expr, normalized_text})
            continue
        skipped_expressions.discard(expr)
        filtered_raw_weights[expr] = weight
        normalized_weight_map[normalized_text] += weight

    if not filtered_raw_weights:
        return None, list(skipped_expressions)

    raw_weight_map = filtered_raw_weights

    finetune_dir = Path(cfg.llm_output_dir) / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = finetune_dir / "dataset.pt"
    scores_csv_path = finetune_dir / "scores.csv"


    dataset = build_dataset(list(raw_weight_map.keys()), max_len=cfg.max_len)
    save_dataset_pt(dataset_path, dataset)


    available_weight_map: Dict[str, float] = {}
    for expr in dataset.expressions:
        available_weight_map[expr] = normalized_weight_map.get(expr, 1.0)

    scores = score_with_llm(
        expressions=dataset.expressions,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=getattr(cfg, 'llm_base_url', None),
        batch_size=getattr(cfg, 'llm_score_batch_size', 5),
        temperature=getattr(cfg, 'llm_temperature', 0.5),
    )

    scored_rows: List[Dict[str, object]] = []
    for item in scores:
        expr = item.get("expression")
        if not isinstance(expr, str):
            continue
        weight_val = float(available_weight_map.get(expr, 1.0))
        scored_rows.append({"expression": expr, "score": item.get("score", 0.0), "weight": weight_val})

    if not scored_rows:
        return None, list(skipped_expressions)
    write_scores_csv(scores_csv_path, scored_rows)

    base_checkpoint_path = Path(cfg.expression_scorer_path)
    finetuned_checkpoint_path = finetune_dir / "expression_transformer.pt"
    init_checkpoint_path = finetuned_checkpoint_path if finetuned_checkpoint_path.is_file() else base_checkpoint_path
    model_cfg: Dict[str, object] = {}
    if init_checkpoint_path.is_file():
        payload = torch.load(init_checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            model_cfg = payload.get("config", {}) or {}

    train_cfg = ExpressionTrainingConfig(
        data_pt=dataset_path,
        scores_csv=scores_csv_path,
        save_path=finetuned_checkpoint_path,
        device=cfg.device,
        seed=getattr(cfg, 'seed', 42),
        batch_size=getattr(cfg, 'llm_ft_batch_size', 8),
        num_epochs=getattr(cfg, 'llm_ft_epochs', 5),
        learning_rate=getattr(cfg, 'llm_ft_lr', 1e-4),
        weight_decay=getattr(cfg, 'llm_ft_weight_decay', 1e-4),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        ff_multiplier=int(model_cfg.get("ff_multiplier", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        grad_clip=getattr(cfg, 'llm_ft_grad_clip', 1.0),
        val_ratio=getattr(cfg, 'llm_ft_val_ratio', 0.1),
        test_ratio=getattr(cfg, 'llm_ft_test_ratio', 0.0),
        save_metadata=False,
        init_checkpoint=init_checkpoint_path if init_checkpoint_path.is_file() else None,
    )

    run_expression_transformer_training(train_cfg)
    torch.cuda.empty_cache()
    if not hasattr(cfg, 'base_expression_scorer_path'):
        cfg.base_expression_scorer_path = base_checkpoint_path
    cfg.latest_expression_scorer_path = finetuned_checkpoint_path
    return ExpressionScorePredictor(finetuned_checkpoint_path, cfg.device), list(skipped_expressions)

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def train_net_p_with_weight(cfg,net,x,y,weights,lr=0.001):
    
    x_train, x_valid, y_train, y_valid,weights_train,weights_valid = train_test_split(x, y,weights, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train,weights_train),
                                batch_size=cfg.batch_size_p, shuffle=True,
                            )
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid,weights_valid), 
                              batch_size=cfg.batch_size_p, shuffle=False)
    
    # loss with weight
    def weighted_mse_loss(input, target, weights):
        out = (input - target)**2
        out = out * weights.expand_as(out)
        loss = out.mean()
        return loss

    # Create your loss function, and optimizer
    # loss_fn = torch.nn.MSELoss()
    loss_fn = weighted_mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    train_regression_model_with_weight(train_loader, valid_loader, net, 
                           loss_fn, optimizer, device=cfg.device,
                           num_epochs=cfg.num_epochs_p, use_tensorboard=False, 
                           tensorboard_path='logs', early_stopping_patience=cfg.es_p)

def get_metric(
    zoo_blds,
    device,
    corr_thresh=0.5,
    metric_target='ic_sharpe',
    ic_weight: float = 0.8,
    robust_evaluator: Optional[NoiseRobustnessEvaluator] = None,
):

    n_blds = len(zoo_blds)
    if n_blds >0:
        n_days = len(zoo_blds.ret_list[0])
        existed = zoo_blds.ret_list # [blds1,blds2,...] ,blds1: (n_days,)
        existed = np.vstack(existed) # (n_blds,n_days)
        existed = torch.from_numpy(existed).to(device)
        assert existed.shape == (n_blds,n_days)
        print(f"existed n_blds == {n_blds}")
    else:
        print("n_blds == 0")
    
    def invalid_to_zero(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(x, 0.0)

    def calc_multi_score(fct, tgt):
        ret = batch_ret(fct,tgt)
        ic = batch_pearsonr(fct,tgt)

        ic_mean = ic.mean().abs().item()
        icir = (ic_mean/ic.std()).item()
        ret_mean = ret.mean().item()
        ret_ir = (ret_mean/ret.std()).item()
        sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()
        ic_sharpe = invalid_to_zero(ic_weight * ic_mean + (1 - ic_weight) * (sharpe / 10))

        multi_score = {
            'ic': invalid_to_zero(ic_mean),
            'icir': invalid_to_zero(icir),
            'ret': invalid_to_zero(ret_mean),
            'sharpe': invalid_to_zero(sharpe),
            'retir': invalid_to_zero(ret_ir),
            'ic_sharpe': ic_sharpe,
        }
        return multi_score, ret, ic

    def get_score(fct, tgt):
        multi_score, ret, _ = calc_multi_score(fct, tgt)
        default_score = multi_score.get('ic_sharpe', multi_score.get('ic', 0.0))
        score = float(multi_score.get(metric_target, default_score))
        
        #  too many nan
        if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() <0.8:
            score = 0.
        
        # unique ratio too small
        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) <0.01:
            score = 0.
        
        if n_blds > 0 and score > 0.:
            assert len(ret.shape) == 1 , f"{ret.shape},{n_days}"
            assert len(ret) == n_days , f"{ret.shape},{n_days}"

            all_matrix = torch.concatenate([existed,ret[None]],dim=0) # (n_blds+1,n_days)
            assert all_matrix.shape == (n_blds+1,n_days) , f"{all_matrix.shape}"

            corr_score = torch.corrcoef(all_matrix)[-1,:-1].abs().max().item()

            if corr_score > corr_thresh:
                score = 0.

        if robust_evaluator is not None and score > 0.0:
            score = float(
                robust_evaluator(
                    fct,
                    tgt,
                    score,
                    calc_multi_score,
                    multi_score,
                )
            )

        multi_score['robust_score'] = float(score)

        if not np.isfinite(score):
            score = 0.0

        return {'score':score,'ret':ret.detach().cpu().numpy(),'multi_score':multi_score}
    return get_score


def main(
        instruments: str = "csi300",
        train_start = 2012,
        train_end_year:int = 2022,
        freq:str = 'day',
        seeds:str = '[0,1,2,3,4]',
        cuda:int = 0,
        save_name:str = 'test',
        zoo_size:int = 10,
        corr_thresh:float = 0.7,
        model_name = "LLMs NAME",
        model_base_url = "LLMs URL",
        model_api_key = "API KEY",
        #csi300
        score_thresh:float = 0.095,
        llm_score_thresh:float = 0.09,
        #csi500
        # score_thresh: float = 0.1,
        # llm_score_thresh: float = 0.095,
        icir_thresh:float = 0.1,
):
    if isinstance(seeds,str):
        seeds = eval(seeds)
    assert isinstance(seeds,list)   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
    train_end = train_end_year
    returned = get_data_by_year(
        train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq=freq,
    )
    data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned

    for seed in seeds:
        reseed_everything(seed)
        start_time = time.time()
        class cfg:
            name = f'{save_name}_{instruments}_{train_end}_{seed}'
            # 
            max_len = 20

            batch_size = 256
            potential_size = 100
            n_layers = 2
            d_model = 128
            dropout = 0.2
            num_factors = zoo_size

            # generator configuaration
            num_epochs_g = 200
            g_es_score = 'max' # max mean std combined
            g_es = 10
            g_hidden = 128
            g_lr = 1e-3

            # complexity estimator configuration
            c_hidden = 128
            c_heads = 4
            c_layers = 2
            c_dropout = 0.1
            c_lr = 1e-3


            # predictor configuration
            p_hidden = 128
            p_lr = 1e-3
            es_p = 10
            batch_size_p = 64
            num_epochs_p = 100
            data_keep_p = 20000

            # robust configuration
            metric_target = 'ic_sharpe'
            robust_enable = True
            robust_noise_scale = 0.5
            robust_samples = 5
            robust_penalty_weight = 0.5
            robust_tolerance = 0.1
            robust_metric_target = 'ic_sharpe'
            robust_renormalize = True
            robust_time_stability_target = 0.8
            robust_time_stability_weight = 0.2

            f_corr_thresh = corr_thresh # threshold to penalize the correlation
            f_add_thresh = corr_thresh # threshold to add new exprs to the zoo
            f_score_thresh = score_thresh # threshold to filter exprs in the zoo
            f_multi_score_thresh = {'icir':icir_thresh}


            # loss configuaration
            l_pred = 1.
            l_simi = 10.
            l_simi_thresh = 0.4


            l_potential = 10.
            l_potential_thresh = 0.4
            l_potential_epsilon = 1e-7

            l_entropy = 0

            l_complexity = 3.0

            device = 'cuda:0'
            expression_scorer_path = Path(__file__).resolve().parent / "llm_labeling" / "out" / "checkpoints" / "expression_transformer_100.pt"
            model_tag = Path(expression_scorer_path).stem.split("_")[-1]
            output_root = Path("out") / model_tag / name

            llm_enable = True
            # llm_enable = False
            llm_model = model_name
            llm_base_url = model_base_url
            llm_candidates = 8
            llm_score_threshold = llm_score_thresh
            llm_timeout = 60.0
            llm_output_dir = output_root / "llm"
            llm_api_key: Optional[str] = model_api_key
            llm_score_batch_size = 5
            llm_temperature = 0.5
            llm_finetune_min_records = 16
            llm_finetune_interval = 2
            llm_ft_prev_weight = 1.0
            llm_ft_curr_weight = 2.0
            llm_ft_batch_size = 4
            llm_ft_epochs = 3
            llm_ft_lr = 1e-4
            llm_ft_weight_decay = 1e-4
            llm_ft_grad_clip = 1.0
            llm_ft_val_ratio = 0.1
            llm_ft_test_ratio = 0.1

        print(f"seed:{seed},name:{cfg.name}")
        cfg.seed = seed

        NetG_CLS = NetG_Mamba
        NetP_CLS = NetPMamba

        def random_call(z):
            return z.normal_()


        netG = NetG_CLS(
            n_chars=SIZE_ACTION,
            latent_size=cfg.potential_size,
            seq_len=cfg.max_len,
            d_model=cfg.g_hidden,
            n_layers=cfg.n_layers,
            ).to(cfg.device)

        netM = NetM(max_len=cfg.max_len,size_action=SIZE_ACTION).to(cfg.device)

        netP = NetP_CLS(
            n_chars=SIZE_ACTION,
            seq_len=cfg.max_len,
            hidden=cfg.p_hidden,
        ).to(cfg.device)

        scorer_path = Path(cfg.expression_scorer_path)
        print(scorer_path)
        if not scorer_path.is_file():
            raise FileNotFoundError(f"Expression scorer checkpoint not found: {scorer_path}")
        expression_scorer = ExpressionScorePredictor(scorer_path, cfg.device)

        z = torch.zeros([cfg.batch_size,cfg.potential_size])
        z = z.to(cfg.device)
        random_call(z)

        llm_agent: Optional[FactorOptimizationAgent] = None
        optimized_cache: Set[str] = set()
        llm_zoo_records: Dict[str, Dict[str, object]] = {}
        llm_output_path = Path(getattr(cfg, 'llm_output_dir'))
        prev_llm_finetune_batch: List[str] = []
        last_llm_finetune_step = -max(1, int(getattr(cfg, 'llm_finetune_interval', 1)))
        pending_finetune_payload: Optional[Tuple[List[str], List[float]]] = None

        if getattr(cfg, 'llm_enable', False):
            try:
                llm_agent = FactorOptimizationAgent(
                    model=cfg.llm_model,
                    base_url=getattr(cfg, 'llm_base_url', 'https://api.deepseek.com'),
                    api_key=getattr(cfg, 'llm_api_key', None),
                    request_timeout=getattr(cfg, 'llm_timeout', 60.0),
                )
            except Exception as exc:
                print(f"Disable LLM optimization: {exc}")
                llm_agent = None

        def llm_goal_met() -> bool:
            if not llm_agent or getattr(cfg, 'llm_candidates', 0) <= 0:
                return True
            return len(llm_zoo_records) >= cfg.num_factors

        robust_eval: Optional[NoiseRobustnessEvaluator] = None
        if getattr(cfg, 'robust_enable', False):
            robust_cfg = NoiseRobustnessConfig(
                noise_scale=cfg.robust_noise_scale,
                num_samples=cfg.robust_samples,
                penalty_weight=cfg.robust_penalty_weight,
                tolerance=cfg.robust_tolerance,
                metric_target=cfg.robust_metric_target,
                renormalize=getattr(cfg, 'robust_renormalize', True),
                time_stability_target=getattr(cfg, 'robust_time_stability_target', 0.7),
                time_stability_weight=getattr(cfg, 'robust_time_stability_weight', 0.0),
            )
            robust_eval = NoiseRobustnessEvaluator(robust_cfg)

        # initialize the zoo
        zoo_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(
            zoo_blds,
            device=cfg.device,
            corr_thresh=cfg.f_corr_thresh,
            metric_target=cfg.metric_target,
            robust_evaluator=robust_eval,
        )
        empty_metric = get_metric(
            Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION),
            device=cfg.device,
            corr_thresh=cfg.f_corr_thresh,
            metric_target=cfg.metric_target,
            robust_evaluator=robust_eval,
        )

        coll = Collector(seq_len=cfg.max_len,n_actions=SIZE_ACTION)
        coll.reset(data,target,metric)
        coll.collect_target_num(netG,netM,z,data,target,metric,
                                target_num=10000,reset_net=True,drop_invalid=False,
                                randomly = False,
                                random_method = random_call,max_iter = 200)


        # train and mine untill the zoo is full
        t = 0
        while len(zoo_blds) < cfg.num_factors or not llm_goal_met():
            previous_batch_for_combo = list(prev_llm_finetune_batch)
            this_iter_finetune_batch: List[str] = []
            if not zoo_blds.examined:
                print(' zoo_blds not examined')
                zoo_blds.evaluate(data,target,empty_metric,verbose=True)

            ### update the metric for the current zoo
            metric = get_metric(
                zoo_blds,
                device=cfg.device,
                corr_thresh=cfg.f_corr_thresh,
                metric_target=cfg.metric_target,
                robust_evaluator=robust_eval,
            )

            ### Prepare data to train predictor
            coll.blds.evaluate(data,target,metric,verbose=True)
            if coll.blds_bak.batch_size>cfg.data_keep_p:
                # sample the training data of predictor to keep the size
                print(f'sample datas to keep {coll.blds_bak.batch_size}to{cfg.data_keep_p}')
                indices = np.random.choice(np.arange(coll.blds_bak.batch_size),cfg.data_keep_p,replace=False)
                coll.blds_bak = coll.blds_bak.filter_by_index(indices)
            coll.blds_bak.evaluate(data,target,metric,verbose=True)



            if coll.blds_bak.batch_size > 0:
                # give the current builders more weights in training p
                blds_list = [coll.blds_bak,coll.blds]
                weight_list = [1.,2.]
            else:
                blds_list = [coll.blds]
                weight_list = [1.]

            x, y, weights = blds_list_to_tensor(blds_list,weight_list)
            y = pre_process_y(y)


            ### train predictor
            netP.initialize_parameters() 
            train_net_p_with_weight(cfg,netP,x,y,weights,lr=cfg.p_lr)

            ### train generator
            netG.initialize_parameters()
            blds_in_train = train_network_generator(
                netG,
                netM,
                netP,
                expression_scorer,
                cfg,
                data,
                target,
                t,
                random_method=random_call,
                metric=metric,
                lr=cfg.g_lr,
                n_actions=SIZE_ACTION,
            )
            
            ### Generate new alpha factors from current Generator
            coll.reset(data,target,metric)
            coll.collect_target_num(netG,netM,z,data,target,metric,
                                    target_num=1000,reset_net=False,drop_invalid=False,
                                    randomly = False,
                                    random_method = random_call,max_iter = 100)

            lengh_s = {"train":len(blds_in_train)}
            lengh_s['new']=len(coll.blds)
            coll.blds = coll.blds + blds_in_train
            coll.blds.drop_duplicated()
            lengh_s['all_new']=len(coll.blds)

            print(f"{lengh_s['train']} (train) + {lengh_s['new']} (new)  =  {lengh_s['all_new']} (all_new)")
            
            ### get the valid alpha factors during the training process and the generating process
            new_zoo = filter_valid_blds(
                coll.blds,
                corr_thresh=cfg.f_add_thresh,
                score_thresh=cfg.f_score_thresh,
                multi_score_thresh = cfg.f_multi_score_thresh,
                device = cfg.device,
                verbose= True,
                )

            accepted_indices = list(range(new_zoo.batch_size))

            if llm_agent and getattr(cfg, 'llm_candidates', 0) > 0 and new_zoo.batch_size > 0:
                try:
                    new_zoo.build_exprs()
                except Exception as exc:
                    print(f"Failed to materialize expressions for LLM optimization: {exc}")
                else:
                    try:
                        target_tensor = normalize_by_day(target.evaluate(data))
                    except Exception as exc:
                        print(f"Unable to evaluate target for LLM scoring: {exc}")
                        target_tensor = None

                    accepted_indices = []
                    for idx, expr_str in enumerate(new_zoo.exprs_str):
                        if expr_str is None or expr_str in optimized_cache:
                            if expr_str in optimized_cache and expr_str in llm_zoo_records:
                                accepted_indices.append(idx)
                            continue
                        try:
                            result = llm_agent.optimize(expr_str, cfg.llm_candidates)
                            result.round_index = t
                            optimized_cache.add(expr_str)

                            candidate_scores: Dict[str, float] = {}
                            best_key: Optional[str] = None
                            best_score: float = float('-inf')
                            scored_candidates: List[str] = []

                            for key, normalized_expr in result.normalized_candidates.items():
                                try:
                                    expr_obj = llm_agent.parse_expression(normalized_expr)
                                    try:
                                        action_length = _expression_action_length(expr_obj)
                                    except TypeError as len_exc:
                                        print(f"LLM candidate '{key}' skipped: {len_exc}")
                                        continue
                                    if action_length > cfg.max_len:
                                        print(
                                            f"LLM candidate '{key}' skipped: length {action_length} exceeds max {cfg.max_len}."
                                        )
                                        continue
                                    invalid_windows = _find_invalid_delta_times(expr_obj, SUPPORTED_DELTA_TIMES)
                                    if invalid_windows:
                                        print(
                                            f"LLM candidate '{key}' skipped: unsupported window sizes {invalid_windows}."
                                        )
                                        continue
                                    invalid_constants = _find_invalid_constants(expr_obj, SUPPORTED_CONSTANTS)
                                    if invalid_constants:
                                        print(
                                            f"LLM candidate '{key}' skipped: unsupported constants {invalid_constants}."
                                        )
                                        continue
                                    try:
                                        expression_to_actions(expr_obj, cfg.max_len)
                                    except Exception as token_exc:
                                        print(
                                            f"LLM candidate '{key}' skipped: token conversion failed - {token_exc}"
                                        )
                                        continue
                                    factor_tensor = expr_obj.evaluate(data)
                                    factor_tensor = normalize_by_day(factor_tensor)
                                    if target_tensor is None:
                                        score_result = {'score': 0.0}
                                    else:
                                        score_result = metric(fct=factor_tensor, tgt=target_tensor)
                                    score_value = float(score_result.get('score', 0.0))
                                except OutOfDataRangeError:
                                    score_value = 0.0
                                except Exception as expr_exc:
                                    print(f"Failed to score LLM candidate '{key}' for factor '{expr_str}': {expr_exc}")
                                    score_value = 0.0

                                candidate_scores[key] = score_value
                                scored_candidates.append(normalized_expr)
                                if score_value > best_score:
                                    best_score = score_value
                                    best_key = key

                            if scored_candidates:
                                this_iter_finetune_batch.extend(scored_candidates)

                            if best_key is not None and best_score >= getattr(cfg, 'llm_score_threshold', cfg.f_score_thresh):
                                normalized_expression = result.normalized_candidates[best_key]
                                record = {
                                    "original_factor": result.original_factor,
                                    "normalized_expression": normalized_expression,
                                    "score": best_score,
                                }
                                existing = llm_zoo_records.get(result.original_factor)
                                if existing is None or existing.get("score", 0.0) < best_score:
                                    llm_zoo_records[result.original_factor] = record
                                    save_llm_factors(list(llm_zoo_records.values()), llm_output_path)
                                accepted_indices.append(idx)
                            else:
                                print(f"No LLM candidate above threshold for factor '{expr_str}'.")
                        except Exception as exc:
                            print(f"LLM optimization failed for factor '{expr_str}': {exc}")
                    if accepted_indices:
                        new_zoo = new_zoo.filter_by_index(accepted_indices)
                    else:
                        new_zoo = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
            if this_iter_finetune_batch:
                curr_weight = float(getattr(cfg, 'llm_ft_curr_weight', 2.0))
                prev_weight = float(getattr(cfg, 'llm_ft_prev_weight', 1.0))
                combined_pairs: List[Tuple[str, float]] = [(expr, curr_weight) for expr in this_iter_finetune_batch]
                if previous_batch_for_combo:
                    combined_pairs.extend((expr, prev_weight) for expr in previous_batch_for_combo)
                min_records = float(getattr(cfg, 'llm_finetune_min_records', 8))
                interval = max(1, int(getattr(cfg, 'llm_finetune_interval', 1)))
                total_weight = sum(weight for _, weight in combined_pairs)
                if total_weight >= min_records:
                    if (t - last_llm_finetune_step) >= interval:
                        expressions_payload = [expr for expr, _ in combined_pairs]
                        weight_payload = [weight for _, weight in combined_pairs]
                        try:
                            finetuned, rejected_exprs = finetune_expression_transformer(cfg, expressions_payload, weight_payload)
                            rejected_set = set(rejected_exprs)
                            if rejected_set:
                                combined_pairs = [(expr, weight) for expr, weight in combined_pairs if expr not in rejected_set]
                                this_iter_finetune_batch = [expr for expr in this_iter_finetune_batch if expr not in rejected_set]
                                previous_batch_for_combo = [expr for expr in previous_batch_for_combo if expr not in rejected_set]
                            if finetuned is not None:
                                expression_scorer = finetuned
                                last_llm_finetune_step = t
                                pending_finetune_payload = None
                            else:
                                remaining_exprs = [expr for expr, _ in combined_pairs]
                                remaining_weights = [weight for _, weight in combined_pairs]
                                if remaining_exprs:
                                    pending_finetune_payload = (remaining_exprs, remaining_weights)
                                else:
                                    pending_finetune_payload = None
                        except Exception as ft_exc:
                            print(f"ExpressionTransformer fine-tune failed: {ft_exc}")
                            pending_finetune_payload = None
                    else:
                        payload_exprs = [expr for expr, _ in combined_pairs]
                        payload_weights = [weight for _, weight in combined_pairs]
                        if payload_exprs:
                            pending_finetune_payload = (payload_exprs, payload_weights)
                        else:
                            pending_finetune_payload = None
                else:
                    pending_finetune_payload = None
            prev_llm_finetune_batch = list(this_iter_finetune_batch) if this_iter_finetune_batch else []
            lengh_s['zoo_prev'] = len(zoo_blds)
            zoo_blds = zoo_blds + new_zoo

            # if len(zoo_blds) > cfg.num_factors:
            #     zoo_blds = zoo_blds.filter_by_index(list(range(cfg.num_factors)))

            print(f" zoo_prev:{lengh_s['zoo_prev']},all_new:{len(new_zoo)},current:{len(zoo_blds)}")
            zoo_blds.evaluate(data,target,empty_metric,verbose=True)
            if t % 5 == 2:
                print('#'*20,"zoo_rebalance")
                zoo_blds = filter_valid_blds(
                    zoo_blds,
                    corr_thresh=cfg.f_add_thresh,
                    score_thresh=cfg.f_score_thresh,
                    multi_score_thresh = cfg.f_multi_score_thresh,
                    device = cfg.device,
                    verbose = False,
                    )
            # save the zoo
            save_blds(zoo_blds,str(cfg.output_root),f'zoo_final_{train_start}_{zoo_size}close')

            if len(zoo_blds) >= cfg.num_factors and llm_goal_met():
                del x,y,weights
                gc.collect()
                torch.cuda.empty_cache()
                break

            # Randomly generate some alpha factors in order to promote exploration and to avoid local minimum
            coll.collect_target_num(netG,netM,z,data,target,metric,
                                    target_num=1000,reset_net=False,drop_invalid=False,
                                    randomly = True,
                                    random_method = random_call,max_iter = 100)

            del x,y,weights
            gc.collect()
            torch.cuda.empty_cache()
            t+=1

        empty_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(
            empty_blds,
            device=cfg.device,
            corr_thresh=cfg.f_corr_thresh,
            metric_target=cfg.metric_target,
            robust_evaluator=robust_eval,
        )
        zoo_blds.evaluate(data,target,metric,verbose=True)
        save_blds(zoo_blds,str(cfg.output_root),f'zoo_final_{train_start}_{zoo_size}close')
        save_llm_factors(list(llm_zoo_records.values()), llm_output_path)
        if pending_finetune_payload:
            expressions_payload, weights_payload = pending_finetune_payload
            try:
                finetuned, rejected_exprs = finetune_expression_transformer(cfg, expressions_payload, weights_payload)
                if rejected_exprs:
                    print(f"ExpressionTransformer fine-tune skipped {len(rejected_exprs)} residual expressions during final step.")
                if finetuned is not None:
                    expression_scorer = finetuned
            except Exception as ft_exc:
                print(f"ExpressionTransformer fine-tune (final) failed: {ft_exc}")
        seed_time = (time.time() - start_time) / 60 / 60
        print(f"{seed} runs {seed_time:.2f} hours and ends")
if __name__ == '__main__':
    import fire
    fire.Fire(main)