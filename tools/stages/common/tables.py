"""The numbers of a table cell.

Each run of a timing block gives one sample for each case. The tables print the median of the samples,
because the phone gives a slow run now and then, and the median is stable against it.
"""

import statistics
from typing import Iterable


def med(values: Iterable[float | None]) -> float | None:
    """The median of the values that are not None, or None when there is no value. Use it for a cell
    that can be empty, and give the result to fmt, which prints a dash for None."""
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def median(values: Iterable[float]) -> float:
    """The median of the values, for a caller that has made sure that there is at least one value. The
    result is never None, thus the caller can calculate with it.

    Raises:
        ValueError: If there is no value. This is an error of the caller, not a condition of the data
    """
    v = list(values)
    if not v:
        raise ValueError("tables.median: no value. Use tables.med for a cell that can be empty.")
    return statistics.median(v)


def fmt(x: float | None, digits: int = 1) -> str:
    """A number with `digits` decimals, or a dash for None."""
    return "-" if x is None else f"{x:.{digits}f}"


def rate(x: float) -> str:
    """A rate with 2 decimals below 100 and 1 decimal from 100, thus each cell has the same width."""
    return f"{x:.2f}" if x < 100 else f"{x:.1f}"


def change(new: float | None, base: float | None, digits: int = 1) -> str:
    """The change of `new` against `base` in percent, or a dash when a value is missing or zero."""
    return "-" if not new or not base else f"{100 * (new / base - 1):+.{digits}f}%"


def span(values: Iterable[float]) -> str:
    """The lowest and the highest value in percent against 1, for a list of ratios."""
    v = list(values)
    return f"[{100 * (min(v) - 1):+.2f}, {100 * (max(v) - 1):+.2f}]" if v else "[]"
