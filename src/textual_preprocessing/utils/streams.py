"""Module for streaming utilities"""
import os
import random
from itertools import islice
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar


def reusable(gen_func: Callable) -> Callable:
    """Function decorator that turns your generator function into an
    iterator, thereby making it reusable.

    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable

    Returns
    -------
    Callable
        Sneakily created iterator class wrapping the generator function
    """

    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit

        def __iter__(self):
            if self.limit is not None:
                return islice(
                    gen_func(*self.__args, **self.__kwargs), self.limit
                )
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


T = TypeVar("T")


@reusable
def chunk(
    iterable: Iterable[T], chunk_size: int, sample_size: Optional[int] = None
) -> Iterable[List[T]]:
    """Generator function that chunks an iterable for you.

    Parameters
    ----------
    iterable: Iterable of T
        The iterable you'd like to chunk.
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want
        those lists to be.

    Yields
    ----------
    buffer: list of T
        sample_size or chunk_size sized lists chunked from
        the original iterable
    """
    buffer = []
    for index, elem in enumerate(iterable):
        buffer.append(elem)
        if (index % chunk_size == (chunk_size - 1)) and (index != 0):
            if sample_size is None:
                yield buffer
            else:
                yield random.choices(buffer, k=sample_size)
            buffer = []


@reusable
def sentence_stream(
    files: List[str], normalize: Callable, lemmatize: Callable
):
    for file in files:
        with open(file) as f:
            text = f.read()
        text = normalize(text)
        sentences = text.split(".")
        for sentence in sentences:
            tokens = sentence.split()
            lemmata = lemmatize(tokens)
            if len(lemmata) >= 5:
                yield lemmata


@reusable
def stream_files(
    paths: Iterable[str], tolerate_failure: bool = True
) -> Iterable[str]:
    """
    Streams file contents from an iterable of file paths.

    Parameters
    -----------
    paths: iterable of str
        The file paths you want to stream
    tolerate_failure: bool, default True
        Specifies whether the stream should tolerate unreadable files.
        If set to True, it will yield an empty string and log the
        file path to stdout instead of raising an exception.

    Yields
    -----------
    contents: str
        Textual content of the current file.
    """
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as in_file:
                contents = in_file.read()
            yield contents
        except FileNotFoundError as e:
            if tolerate_failure:
                yield ""
            else:
                raise e


BAR_LENGTH = 100
N_DECIMALS = 1
FILL_CHARACTER = "█"


@reusable
def progress_bar_stream(
    items: Sequence[T], total: Optional[int] = None
) -> Iterable[T]:
    """
    Wraps list in an iterable that shows a progress bar
    and the current element.

    Parameters
    ----------
    items: iterable of U
        Items to iterate over (of type U)
    total: int or None, default None
        Total number of items to process, if not specified, len(items)
        will be taken as the total.
    Yields
    ----------
    item: U
        Current item under processing
    """
    if total is None:
        total = len(items)
    for iteration, item in enumerate(items):
        percent = ("{0:." + str(N_DECIMALS) + "f}").format(
            100 * (iteration / float(total))
        )
        filled_length = int(BAR_LENGTH * iteration // total)
        progress_bar = FILL_CHARACTER * filled_length + "-" * (
            BAR_LENGTH - filled_length
        )
        os.system("clear")
        print(f"Progress: |{progress_bar}| {percent}%")
        yield item
