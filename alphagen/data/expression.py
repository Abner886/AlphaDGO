from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

import torch
from torch import Tensor

from alphagen_qlib.stock_data import StockData, FeatureType


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["Expression", float]) -> "Sub":
        if isinstance(other, Expression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["Expression", float]) -> "Mul":
        if isinstance(other, Expression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["Expression", float]) -> "Div":
        if isinstance(other, Expression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pow__(self, other: Union["Expression", float]) -> "Pow":
        if isinstance(other, Expression):
            return Pow(self, other)
        else:
            return Pow(self, Constant(other))

    def __rpow__(self, other: float) -> "Pow": return Pow(Constant(other), self)

    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    # def __str__(self) -> str: return '$' + self._feature.name.lower()

    def __str__(self) -> str:  return f"{self._feature.name.lower()}"

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        data_tensor = data.data if torch.is_tensor(data.data) else None
        raw_dtype = data_tensor.dtype if data_tensor is not None else None
        dtype = raw_dtype if isinstance(raw_dtype, torch.dtype) else torch.float32
        days = period.stop - period.start - 1 + data.n_days
        base = torch.empty((days, data.n_stocks), dtype=dtype)
        base.fill_(float(self._value))
        if data_tensor is not None:
            base = base.to(device=data_tensor.device)
        return base

    # def __str__(self) -> str: return f'Constant({str(self._value)})'
    def __str__(self) -> str: return f'{str(self._value)}'

    @property
    def is_featured(self): return False


class Sequence(Expression):
    def __init__(self, length: int) -> None:
        self._length = int(length)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        data_tensor = data.data if torch.is_tensor(data.data) else None
        raw_dtype = data_tensor.dtype if data_tensor is not None else None
        dtype = raw_dtype if isinstance(raw_dtype, torch.dtype) else torch.float32
        days = period.stop - period.start - 1 + data.n_days
        base = torch.arange(1, days + 1, dtype=dtype)
        if data_tensor is not None:
            base = base.to(device=data_tensor.device)
        return base[:, None].expand(days, data.n_stocks)

    def __str__(self) -> str: return f"Sequence({self._length})"

    @property
    def is_featured(self): return True


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return str(self._delta_time)

    @property
    def is_featured(self): return False


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        name = type(self).__name__
        if name == 'Add':
            return f"({self._lhs}+{self._rhs})"
        elif name == 'Sub':
            return f"({self._lhs}-{self._rhs})"
        elif name == 'Mul':
            return f"({self._lhs}*{self._rhs})"
        elif name == 'Div':
            return f"({self._lhs}/{self._rhs})"
        elif name == 'Pow':
            return f"({self._lhs}**{self._rhs})"
        else:
            return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = int(delta_time)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start_idx = 0 if period.start is None else int(period.start)
        stop_idx = start_idx + 1 if period.stop is None else int(period.stop)
        start = start_idx - self._delta_time + 1
        stop = stop_idx
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = int(delta_time)

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: Expression,
                    data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start_idx = 0 if period.start is None else int(period.start)
        stop_idx = start_idx + 1 if period.stop is None else int(period.stop)
        start = start_idx - self._delta_time + 1
        stop = stop_idx
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingConstOperator(Operator):
    def __init__(
        self,
        operand: Union[Expression, float],
        delta_time: Union[int, DeltaTime],
        const: Union[Expression, float],
    ) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = int(delta_time)
        if isinstance(const, Expression):
            if not isinstance(const, Constant):
                raise ValueError("RollingConstOperator expects a constant value")
            self._const = const
        else:
            self._const = Constant(const)

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingConstOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start_idx = 0 if period.start is None else int(period.start)
        stop_idx = start_idx + 1 if period.stop is None else int(period.stop)
        start = start_idx - self._delta_time + 1
        stop = stop_idx
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values, float(self._const._value))       # type: ignore[attr-defined]

    @abstractmethod
    def _apply(self, operand: Tensor, const: float) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time},{self._const})"

    @property
    def is_featured(self): return self._operand.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()

    
class S_log1p(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * operand.abs().log1p()
    
class Inv(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return 1/operand


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class Sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.clamp_min(0).sqrt()


class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class Zscore(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        count = (~nan_mask).sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1)
        centered = operand.masked_fill(nan_mask, 0)
        mean = centered.sum(dim=1, keepdim=True) / safe_count
        diff = operand - mean
        var = diff.masked_fill(nan_mask, 0).pow(2).sum(dim=1, keepdim=True) / safe_count
        std = var.sqrt()
        std = std.masked_fill(std < 1e-6, 1)
        z = diff / std
        z[nan_mask] = torch.nan
        return z


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)

class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)

class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1, unbiased=False)

class ts_Zscore(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        mean = operand.mean(dim=-1)
        std = operand.std(dim=-1, unbiased=False)
        std = std.masked_fill(std < 1e-6, 1)
        return (operand[..., -1] - mean) / std

class Ir(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1) / operand.std(dim=-1, unbiased=False)

class Min_max_diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
class Max_diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.max(dim=-1)[0]
class Min_diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.min(dim=-1)[0]

class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1, unbiased=False)

class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5

class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1, unbiased=False)
        return m4 / var ** 2 - 3

class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]

class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]

class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]

class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)

class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class ts_div(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] / values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class Pctchange(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] / values[:-self._delta_time] - 1

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class Wma(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Decaylinear(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(1, n + 1, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class  Ema(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class ts_quantile(RollingConstOperator):
    def _apply(self, operand: Tensor, const: float) -> Tensor:
        q = min(max(const, 0.0), 1.0)
        return torch.quantile(operand, q, dim=-1)


class Percentile(RollingConstOperator):
    def __init__(
        self,
        operand: Union[Expression, float],
        arg2: Union[Expression, float, int, DeltaTime],
        arg3: Union[Expression, float, int, DeltaTime],
    ) -> None:
        if isinstance(arg2, (DeltaTime, int)):
            delta_time = arg2
            const = arg3
        else:
            const = arg2
            delta_time = arg3
        super().__init__(operand, delta_time, const)

    def _apply(self, operand: Tensor, const: float) -> Tensor:
        q = min(max(const, 0.0), 1.0)
        return torch.quantile(operand, q, dim=-1)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul


class Regresi(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        mean_lhs = lhs.mean(dim=-1)
        mean_rhs = rhs.mean(dim=-1)
        clhs = lhs - mean_lhs[..., None]
        crhs = rhs - mean_rhs[..., None]
        cov = (clhs * crhs).sum(dim=-1)
        var = (crhs ** 2).sum(dim=-1)
        safe_var = var.masked_fill(var < 1e-6, 1)
        beta = cov / safe_var
        beta = beta.masked_fill(var < 1e-6, 0)
        alpha = mean_lhs - beta * mean_rhs
        pred = alpha + beta * rhs[..., -1]
        return lhs[..., -1] - pred


Operators: List[Type[Expression]] =  [
    # Unary
    Abs,
    Sign,
    Log,
    Sqrt,
    Inv,
    S_log1p,
    # CSRank,
    Zscore,
    Pow,
    # Binary
    Add, Sub, Mul, Div,
    # Greater, Less,

    # Rolling
    Ref, Mean, Sum, Std, Var,
    ts_Zscore,
    Skew,
    Kurt,
    Max, Min,
    Med, Mad,  Rank,
    ts_div,
    Pctchange,
    Ir,
    Min_max_diff,
    Max_diff,Min_diff,
    Delta,
    # Delta,
    Wma,
    Decaylinear,
    Ema,
    ts_quantile,
    Percentile,
    # WMA, EMA,

    # Pair rolling
    Cov, Corr,
    Regresi
]