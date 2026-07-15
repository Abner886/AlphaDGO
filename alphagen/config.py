from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS = Operators
OPERATORS = [
    # Unary
    Abs,
    # Sign,
    Log,
    Inv,
    S_log1p,
    # CSRank,
    # Binary,
    Add, Sub, Mul, Div, 
    Pow,
    # Greater, Less,

    # Rolling
    Ref, Mean, Sum, Std, Var,
    Skew,
    Kurt,
    Max, Min,
    Med, Mad,
    # ts_rank,

    ts_div,
    Pctchange,
    Ir,
    # ts_min_max_diff,
    # ts_max_diff,ts_min_diff,
    Delta, Wma, Ema,
    # ts_quantile,

    # Pair rolling
    Cov, Corr
]

DELTA_TIMES = [1, 5, 10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
