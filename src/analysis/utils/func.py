"""
Utilities for functional programming that are not present in Python by default.
"""

from typing import Callable, TypeVar, Iterable
import itertools

# Sorry I'm a Haskell boy :((
T = TypeVar("T")
U = TypeVar("U")


def flatten(iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Flattens an iterable of iterables.
    """
    return itertools.chain.from_iterable(iterable)


def flatmap(
    func: Callable[[T], Iterable[U]], *iterable: Iterable[T]
) -> Iterable[U]:
    """
    Combination of map and flatten operations
    """
    return flatten(map(func, *iterable))
