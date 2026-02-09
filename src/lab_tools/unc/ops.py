from __future__ import annotations
from functools import wraps
import inspect
import itertools
import numbers
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator
import numpy as np
from math import log, sqrt
import sys

if TYPE_CHECKING:
    from .core import AffineScalarFunc

FLOAT_LIKE_TYPES: tuple[type, ...] = (
    numbers.Number,
    np.generic,
)
CONSTANT_TYPES: tuple[type, ...] = FLOAT_LIKE_TYPES + (complex,)


def set_doc(doc_string: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        func.__doc__ = doc_string
        return func

    return decorator


def nan_if_exception(f: Callable) -> Callable:
    @wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> float:
        try:
            return f(*args, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")

    return wrapped


def pow_deriv_0(x: float, y: float) -> float:
    if x > 0 or (y % 1 == 0 and (x < 0 or y >= 1)):
        return y * x ** (y - 1)
    if x == 0 and y == 0:
        return 0.0
    return float("nan")


def pow_deriv_1(x: float, y: float) -> float:
    if x > 0:
        return log(x) * x**y
    if x == 0 and y > 0:
        return 0.0
    return float("nan")


def get_ops_with_reflection() -> dict[str, list[Callable]]:
    derivatives_list = {
        "add": ("1.", "1."),
        "div": ("1/y", "-x/y**2"),
        "floordiv": ("0.", "0."),
        "mod": ("1.", "partial_derivative(float.__mod__, 1)(x, y)"),
        "mul": ("y", "x"),
        "sub": ("1.", "-1."),
        "truediv": ("1/y", "-x/y**2"),
    }
    ops: dict[str, list[Callable]] = {}

    for op, derivatives in derivatives_list.items():
        funcs = [eval(f"lambda x, y: {expr}") for expr in derivatives]
        ops[op] = funcs
        ops[f"r{op}"] = [eval(f"lambda y, x: {expr}") for expr in reversed(derivatives)]

    ops["pow"] = [pow_deriv_0, pow_deriv_1]
    ops["rpow"] = [
        lambda y, x: pow_deriv_1(x, y),
        lambda y, x: pow_deriv_0(x, y),
    ]

    for op in ["pow"]:
        ops[op] = [nan_if_exception(f) for f in ops[op]]
        ops[f"r{op}"] = [nan_if_exception(f) for f in ops[f"r{op}"]]

    return ops


ops_with_reflection = get_ops_with_reflection()
modified_operators: list[str] = []
modified_ops_with_reflection: list[str] = []


def no_complex_result(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> float:
        value = func(*args, **kwargs)
        if isinstance(value, complex):
            raise ValueError("Complex results are not supported")
        return value

    return wrapper


custom_ops = {
    "pow": no_complex_result(float.__pow__),
    "rpow": no_complex_result(float.__rpow__),
}


def add_arithmetic_ops(cls: type) -> None:
    """Add arithmetic operator overloads to AffineScalarFunc-like class."""

    def _simple_add_deriv(x: float) -> float:
        return 1.0 if x >= 0 else -1.0

    simple_ops: dict[str, Callable[[float], float]] = {
        "abs": _simple_add_deriv,
        "neg": lambda x: -1.0,
        "pos": lambda x: 1.0,
        "trunc": lambda x: 0.0,
    }

    for op, derivative in simple_ops.items():
        attr = f"__{op}__"
        try:
            float_impl = getattr(float, attr)
        except AttributeError:
            continue

        setattr(cls, attr, _wrap(cls, float_impl, [derivative]))
        modified_operators.append(op)

        for op, derivatives in ops_with_reflection.items():
            attr = f"__{op}__"

            try:
                func: Callable[..., Any] | None
                if op not in custom_ops:
                    func = getattr(float, attr, None)
                else:
                    func = custom_ops.get(op)
            except AttributeError:
                continue

            if func is None:
                continue

            setattr(cls, attr, _wrap(cls, func, derivatives))
            modified_ops_with_reflection.append(op)

    def raise_error(self):
        raise TypeError(
            f"can't convert affine function ({self.__class__}) "
            f"to numeric type; use x.nominal_value"
        )

    for coercion_type in ("complex", "int", "float"):
        setattr(cls, f"__{coercion_type}__", raise_error)


class IndexableIter:
    def __init__(
        self,
        iterable: Iterable,
        none_converter: Callable[[int], Any] | None = None,
    ) -> None:
        self.iterable: Iterator = iter(iterable)
        self.returned_elements: list[Any] = []
        self.none_converter = none_converter or (lambda _: None)

    def __getitem__(self, index: int) -> Any:
        try:
            return self.returned_elements[index]
        except IndexError:
            for pos in range(len(self.returned_elements), index + 1):
                value = next(self.iterable)
                if value is None:
                    value = self.none_converter(pos)
                self.returned_elements.append(value)
            return self.returned_elements[index]

    def __str__(self) -> str:
        cached = ", ".join(map(str, self.returned_elements))
        return f"<{self.__class__.__name__}: [{cached}...]>"


STEP_SIZE = sqrt(sys.float_info.epsilon)


def partial_derivative(f: Callable, arg_ref: int | str) -> Callable:

    if isinstance(arg_ref, str):

        def derivative(*args: Any, **kwargs: Any) -> float:
            args_with_var = dict(kwargs)

            step = STEP_SIZE * abs(args_with_var[arg_ref])
            if not step:
                step = STEP_SIZE

            args_with_var[arg_ref] += step
            shifted_plus = f(*args, **args_with_var)

            args_with_var[arg_ref] -= 2 * step
            shifted_minus = f(*args, **args_with_var)

            return (shifted_plus - shifted_minus) / (2 * step)

    else:

        def derivative(*args: Any, **kwargs: Any) -> float:
            args_with_var = list(args)

            step = STEP_SIZE * abs(args_with_var[arg_ref])
            if not step:
                step = STEP_SIZE

            args_with_var[arg_ref] += step
            shifted_plus = f(*args_with_var, **kwargs)

            args_with_var[arg_ref] -= 2 * step
            shifted_minus = f(*args_with_var, **kwargs)

            return (shifted_plus - shifted_minus) / (2 * step)

    return derivative


def eq_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    difference: AffineScalarFunc = self - y
    return not (difference._nominal_value or difference.std_dev)


def ne_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    return not eq_on_aff_funcs(self, y)


def gt_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    return self._nominal_value > y._nominal_value


def ge_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    return gt_on_aff_funcs(self, y) or eq_on_aff_funcs(self, y)


def lt_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    return self._nominal_value < y._nominal_value


def le_on_aff_funcs(self: AffineScalarFunc, y: AffineScalarFunc):
    return lt_on_aff_funcs(self, y) or eq_on_aff_funcs(self, y)


class NotUpcast(Exception):
    """Raised when an object cannot be converted to an affine scalar."""


def _wrap(
    cls: type,
    f: Callable[..., Any],
    derivatives_args: Iterable[Callable[..., float] | None] | None = None,
    derivatives_kwargs: dict[str, Callable[..., float] | None] | None = None,
) -> Callable[..., Any]:

    derivatives_args_list = list(derivatives_args or [])
    derivatives_kwargs = derivatives_kwargs or {}

    derivatives_args_index = IndexableIter(
        itertools.chain(derivatives_args_list, itertools.repeat(None))
    )

    derivatives_all_kwargs: dict[str, Callable[..., float]] = {}

    for name, derivative in derivatives_kwargs.items():
        derivatives_all_kwargs[name] = (
            partial_derivative(f, name) if derivative is None else derivative
        )

    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        sig = None

    if sig:
        for index, (name, param) in enumerate(sig.parameters.items()):
            if param.kind not in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            ):
                continue

            derivative = derivatives_args_index[index]

            derivatives_all_kwargs[name] = (
                partial_derivative(f, name) if derivative is None else derivative
            )

    def none_converter(index: int) -> Callable[..., float]:
        return partial_derivative(f, index)

    for i, deriv in enumerate(derivatives_args_index.returned_elements):
        if deriv is None:
            derivatives_args_index.returned_elements[i] = none_converter(i)

    derivatives_args_index.none_converter = none_converter

    @wraps(f)
    def f_with_affine_output(
        *args: AffineScalarFunc,
        **kwargs: AffineScalarFunc,
    ) -> Any:

        pos_w_unc = [i for i, v in enumerate(args) if isinstance(v, cls)]
        names_w_unc = [k for k, v in kwargs.items() if isinstance(v, cls)]

        if not pos_w_unc and not names_w_unc:
            return f(*args, **kwargs)

        args_values: list[float] = [0] * len(args)

        for i in pos_w_unc:
            args_values[i] = args[i].nominal_value

        kwargs_unc_values: dict[str, AffineScalarFunc] = {}
        kwargs_values: dict[str, float] = {}

        for name in names_w_unc:
            val = kwargs[name]
            assert isinstance(val, AffineScalarFunc)
            kwargs_unc_values[name] = val
            kwargs_values[name] = val.nominal_value

        f_nominal_value = f(*args_values, **kwargs_values)

        if not isinstance(f_nominal_value, FLOAT_LIKE_TYPES):
            return NotImplemented

        linear_part: list[tuple[float, Any]] = []

        for pos in pos_w_unc:
            linear_part.append(
                (
                    derivatives_args_index[pos](*args_values, **kwargs_values),
                    args[pos]._linear_part,
                )
            )

        for name in names_w_unc:
            derivative = derivatives_all_kwargs.setdefault(
                name,
                partial_derivative(f, name),
            )

            linear_part.append(
                (
                    derivative(*args_values, **kwargs_values),
                    kwargs_unc_values[name]._linear_part,
                )
            )

        return cls(f_nominal_value, linear_part)

    setattr(f_with_affine_output, "name", f.__name__)

    return f_with_affine_output


def add_comparative_ops(cls):

    def to_affine_scalar(x: Any) -> Any:
        if isinstance(x, cls):
            return x

        if isinstance(x, CONSTANT_TYPES):
            return cls(x, {})

        raise NotUpcast(
            "%s cannot be converted to a number with" " uncertainty" % type(x)
        )

    setattr(cls, "_to_affine_scalar", to_affine_scalar)

    def force_aff_func_args(func):

        def op_on_upcast_args(x, y):

            try:
                y_with_unc = to_affine_scalar(y)
            except NotUpcast:
                return NotImplemented
            else:
                return func(x, y_with_unc)

        return op_on_upcast_args

    def __bool__(self):
        """
        Equivalent to self != 0.
        """
        return self != 0.0

    cls.__bool__ = __bool__
    cls.__eq__ = force_aff_func_args(eq_on_aff_funcs)

    cls.__ne__ = force_aff_func_args(ne_on_aff_funcs)
    cls.__gt__ = force_aff_func_args(gt_on_aff_funcs)
    cls.__ge__ = force_aff_func_args(ge_on_aff_funcs)

    cls.__lt__ = force_aff_func_args(lt_on_aff_funcs)
    cls.__le__ = force_aff_func_args(le_on_aff_funcs)
