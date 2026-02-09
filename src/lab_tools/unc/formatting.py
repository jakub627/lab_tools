import re
from typing import Literal
import numpy as np

from typing import TYPE_CHECKING

from decimal import Decimal, ROUND_HALF_UP, getcontext

if TYPE_CHECKING:
    from .core import AffineScalarFunc


getcontext().prec = 50


def round_half_up(value: float, digits: int = 0) -> float:
    q = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def first_digit(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 0
    return int(np.floor(np.log10(abs(value))))


def get_precision(std_dev: float) -> tuple[Literal[1, 2], float]:
    exponent: int = first_digit(std_dev)
    if exponent >= 0:
        (exponent, factor) = (exponent - 2, 1)
    else:
        (exponent, factor) = (exponent + 1, 1000)
    return (2, 10.0**exponent * (1000 / factor))


robust_format = format
EXP_LETTERS = {"f": "e", "F": "E"}


def robust_align(orig_str: str, fill_char: str, align_option: str, width: str):
    return format(orig_str, fill_char + align_option + width)


TO_SUPERSCRIPT = {
    0x2B: "⁺",
    0x2D: "⁻",
    0x30: "⁰",
    0x31: "¹",
    0x32: "²",
    0x33: "³",
    0x34: "⁴",
    0x35: "⁵",
    0x36: "⁶",
    0x37: "⁷",
    0x38: "⁸",
    0x39: "⁹",
}
FROM_SUPERSCRIPT = {ord(sup): normal for (normal, sup) in TO_SUPERSCRIPT.items()}


def to_superscript(value: int):
    return (f"{value:d}").translate(TO_SUPERSCRIPT)


def normalize_superscript(number_str: str) -> int:
    return int(number_str.translate(FROM_SUPERSCRIPT))


PM_SYMBOLS = {"pretty-print": "±", "latex": r" \pm ", "default": "+/-"}
MULTIPLICATION_SYMBOLS = {"pretty-print": "×", "latex": r"\times"}
EXP_PRINT = {
    "pretty-print": lambda common_exp: f"{MULTIPLICATION_SYMBOLS["pretty-print"]}10{to_superscript(common_exp)}",
    "latex": lambda common_exp: rf" {MULTIPLICATION_SYMBOLS['latex']} 10^{{{common_exp}}}",
}
GROUP_SYMBOLS = {
    "pretty-print": ("(", ")"),
    "latex": (r"\left(", r"\right)"),
    "default": ("(", ")"),
}




def format_num(
    *,
    nom_val_main: float,
    error_main: float,
    fmt_parts: dict[str, str],
    precision: int,
    main_pres_type: Literal["f", "F"],
    options: set[str],
    common_exp: int | None = None,
) -> str:
    if "P" in options:
        print_type = "pretty-print"
    elif "L" in options:
        print_type = "latex"
    else:
        print_type = "default"

    if common_exp is None:
        exp_str = ""
    elif print_type == "default":
        exp_str = f"{EXP_LETTERS[main_pres_type]}{common_exp:+03d}"
    else:
        exp_str = EXP_PRINT[print_type](common_exp)

    percent_str = ""
    if "%" in options:
        percent_str = (" \\" if print_type == "latex" else "") + "%"

    def rounded(value: float) -> float:
        if not np.isfinite(value):
            return value
        return round_half_up(value, precision)

    nom_val = rounded(nom_val_main)
    err_val = rounded(error_main) if error_main else error_main

    special_error = not error_main or not np.isfinite(error_main)

    if special_error and fmt_parts["type"] in ("", "g", "G"):
        fmt_suffix_n = f"{fmt_parts['prec'] or ''}{fmt_parts['type']}"
    else:
        fmt_suffix_n = f".{precision}{main_pres_type}"

    if "S" in options:
        if error_main == 0:
            unc_str = "0"
        elif np.isnan(error_main):
            unc_str = robust_format(error_main, main_pres_type)
        elif np.isinf(error_main):
            unc_str = (
                r"\infty"
                if print_type == "latex"
                else robust_format(error_main, main_pres_type)
            )
        else:
            if first_digit(err_val) >= 0 and precision > 0:
                unc_str = f"{err_val:.{precision}f}"
            else:
                scaled = round_half_up(err_val * 10**precision)
                unc_str = str(int(scaled)) if scaled else "0"

        value_end = f"({unc_str}){exp_str}{percent_str}"

        if fmt_parts["zero"] and fmt_parts["width"]:
            width = max(int(fmt_parts["width"]) - len(value_end), 0)
            fmt_prefix = (
                f"{fmt_parts['sign']}{fmt_parts['zero']}{width}{fmt_parts['comma']}"
            )
        else:
            fmt_prefix = fmt_parts["sign"] + fmt_parts["comma"]

        nom_str = robust_format(nom_val, fmt_prefix + fmt_suffix_n)

        if print_type == "latex":
            if np.isnan(nom_val):
                nom_str = rf"\mathrm{{{nom_str}}}"
            elif np.isinf(nom_val):
                nom_str = rf"{'-' if nom_val < 0 else ''}\infty"

        value = nom_str + value_end

        if fmt_parts["width"]:
            value = robust_align(
                value,
                fmt_parts["fill"],
                fmt_parts["align"] or ">",
                fmt_parts["width"],
            )

        return value

    any_exp_factored = not fmt_parts["width"]
    error_has_exp = not any_exp_factored and not special_error
    nom_has_exp = not any_exp_factored and np.isfinite(nom_val)

    def build_prefix(has_exp: bool) -> str:
        if not fmt_parts["width"]:
            return fmt_parts["sign"] + fmt_parts["comma"]

        if fmt_parts["zero"]:
            width = int(fmt_parts["width"])
            remaining = max(width - len(exp_str), 0)
            return f"{fmt_parts['sign']}{fmt_parts['zero']}{remaining if has_exp else width}{fmt_parts['comma']}"
        return fmt_parts["sign"] + fmt_parts["comma"]

    fmt_prefix_n = build_prefix(nom_has_exp)
    fmt_prefix_e = build_prefix(error_has_exp)

    nom_str = robust_format(nom_val, fmt_prefix_n + fmt_suffix_n)

    if error_main:
        fmt_suffix_e = (
            f"{fmt_parts['prec'] or ''}{fmt_parts['type']}"
            if not np.isfinite(nom_val) and fmt_parts["type"] in ("", "g", "G")
            else f".{precision}{main_pres_type}"
        )
    else:
        fmt_suffix_e = f".0{main_pres_type}"

    err_str = robust_format(err_val, fmt_prefix_e + fmt_suffix_e)

    if print_type == "latex":
        if np.isnan(nom_val):
            nom_str = rf"\mathrm{{{nom_str}}}"
        elif np.isinf(nom_val):
            nom_str = rf"{'-' if nom_val < 0 else ''}\infty"

        if np.isnan(err_val):
            err_str = rf"\mathrm{{{err_str}}}"
        elif np.isinf(err_val):
            err_str = r"\infty"

    if nom_has_exp:
        nom_str += exp_str
    if error_has_exp:
        err_str += exp_str

    if fmt_parts["width"]:
        align = fmt_parts["align"] or ">"
        nom_str = robust_align(nom_str, fmt_parts["fill"], align, fmt_parts["width"])
        err_str = robust_align(err_str, fmt_parts["fill"], align, fmt_parts["width"])

    pm = PM_SYMBOLS[print_type]
    left, right = GROUP_SYMBOLS[print_type]

    if any_exp_factored and common_exp is not None:
        return f"{left}{nom_str}{pm}{err_str}{right}{exp_str}{percent_str}"

    value = f"{nom_str}{pm}{err_str}"
    if percent_str or "p" in options:
        return f"{left}{value}{right}{percent_str}"

    return value


def significant_digits_to_limit(value: float, num_significant_digit: int):
    fst_digit = first_digit(value)
    limit_no_rounding = fst_digit - num_significant_digit + 1
    rounded = round_half_up(value, -limit_no_rounding)
    fst_digit_rounded = first_digit(rounded)

    if fst_digit_rounded > fst_digit:
        limit_no_rounding += 1

    return limit_no_rounding


def format_float(float_to_format: "AffineScalarFunc", format_spec: str):
    match = re.match(
        r"""
        (?P<fill>[^{}]??)(?P<align>[<>=^]?)  # fill cannot be { or }
        (?P<sign>[-+ ]?)
        (?P<zero>0?)
        (?P<width>\d*)
        (?P<comma>,?)
        (?:\.(?P<prec>\d+))?
        (?P<uncert_prec>u?)  # Precision for the uncertainty?
        # The type can be omitted. Options must not go here:
        (?P<type>[eEfFgG%]??)  # n not supported
        (?P<options>[PSLp]*)  # uncertainties-specific flags
        $""",
        format_spec,
        re.VERBOSE,
    )

    if not match:
        raise ValueError(
            f"Format specification {format_spec!r} cannot be used "
            f"with object of type {type(float_to_format).__name__!r}. "
            "Uncertainty-specific flags must be at the end."
        )
    fmt_parts: dict[str, str] = match.groupdict()
    pres_type: str | None = fmt_parts["type"] or None
    fmt_precision: str = fmt_parts["prec"]
    options: set[str] = set(fmt_parts["options"])

    nom_val: float = float_to_format.nominal_value
    std_dev: float = float_to_format.std_dev

    if pres_type == "%":
        std_dev *= 100
        nom_val *= 100
        pres_type = "f"
        options.add("%")

    real_values = [value for value in [abs(nom_val), std_dev] if np.isfinite(value)]

    if (
        ((not fmt_precision and len(real_values) == 2) or fmt_parts.get("uncert_prec"))
        and std_dev
        and np.isfinite(std_dev)
    ):
        num_significant_digit = (
            int(fmt_precision) if fmt_precision else get_precision(std_dev)[0]
        )
        digits_limit = significant_digits_to_limit(std_dev, num_significant_digit)
    else:
        precision = (
            int(fmt_precision) if fmt_precision else (12 if pres_type is None else 6)
        )
        if pres_type in ("f", "F"):
            digits_limit = -precision
        else:
            num_significant_digits = (
                precision + 1 if pres_type in ("e", "E") else max(precision, 1)
            )
            digits_limit = significant_digits_to_limit(
                max(real_values) if real_values else 0, num_significant_digits
            )

    use_exp = pres_type not in ("f", "F") and bool(real_values)
    common_exp: int | None = None
    common_factor: float = 1.0

    if use_exp and real_values:
        ref_val = max(real_values)
        common_exp = first_digit(round_half_up(ref_val, -digits_limit))
        common_factor = 10.0**common_exp

        if pres_type not in ("e", "E") and common_exp is not None:
            use_exp = (
                not (-4 <= common_exp < (-digits_limit + 1)) and common_factor != 0.0
            )
        else:
            use_exp = True

    if use_exp and common_exp is not None:
        nom_val_mantissa = nom_val / common_factor
        std_dev_mantissa = std_dev / common_factor
        significant_limit = digits_limit - common_exp
    else:
        nom_val_mantissa = nom_val
        std_dev_mantissa = std_dev
        significant_limit = digits_limit
        common_exp = None

    main_pres_type = "F" if (pres_type or "g").isupper() else "f"

    precision = max(-significant_limit, 1 if pres_type is None and not std_dev else 0)
    return format_num(
        nom_val_main=nom_val_mantissa,
        error_main=std_dev_mantissa,
        common_exp=common_exp,
        fmt_parts=fmt_parts,
        precision=precision,
        main_pres_type=main_pres_type,
        options=options,
    )
