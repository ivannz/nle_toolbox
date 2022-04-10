import plyr
import torch
import numpy as np

from typing import Any, Union
from torch import Tensor
from numpy import ndarray

from collections import defaultdict, deque


def cat(
    arraylike: Union[Tensor, ndarray],
    dim: int = 0,
) -> Union[Tensor, ndarray]:
    """Concatenate the ndarray or tensor data along the specified dim.
    """
    if isinstance(arraylike[0], Tensor):
        return torch.cat(arraylike, dim=dim)

    return np.concatenate(arraylike, axis=dim)


def stitch(
    *chunks: Any,
    dim: int = 0,
) -> Any:
    """Stitch the structured fragments.
    """
    return plyr.apply(cat, *chunks, _star=False, dim=dim)


def extract(
    strands: dict[int, list],
    reset: Union[Tensor, ndarray],
    fragment: Any,
) -> Any:
    """Generate completed episodes from the given trajectory fragments,
    the corresponding reset mask and incomplete strands.

    Parameters
    ----------
    strands : container of lists
        The incomplete fragmented trajectory traced in each environment.

    reset : boolean array-like, shape=(T, B)
        A boolean mask, indicating if the associated record in `fragment` is
        marked as the end-of-episode. Either a numpy array or a torch tensor.

    fragment : Nested Container of array-like, shape=(T, B, ...)
        The new fragment of trajectories with in each environment. Can be
        a mix of numpy arrays and torch tensors.

    Yields
    ------
    episode : Nested Container of array-like, shape=(L, ...)
        The full trajectory of each episode, completed by the given fragment.
        Unlike the `fragment`, the history DOES NOT have the batch dimension.
    """
    # for each independent env
    for j in range(reset.shape[1]):
        # find all reset brackets [t0, t1)
        t1 = None
        for t, fin in enumerate(reset[:, j]):
            if not fin:
                continue

            # get the [t0, t1+1) slice (we include the `t1` reset, since it
            #  contains the reward and the lethal action).
            t1, t0 = t, t1
            tail = plyr.apply(lambda x: x[t0:t1+1, j], fragment)

            # stitch the fragments together and drop the incomplete strand
            yield stitch(*strands[j], tail)
            strands[j].clear()

        # commit the residual piece [t1, +oo) to the strands
        strands[j].append(plyr.apply(lambda x: x[t1:, j], fragment))


class EpisodeExtractor:
    """A simple object to track and stitch fragmented trajectories.
    """
    def __init__(self) -> None:
        # the fragmented contiguous trajectory of each incomplete episode
        self.strands = defaultdict(list)

    def extract(
        self,
        reset: Union[Tensor, ndarray],
        fragment: Any,
    ) -> list[Any]:
        # the list of completed episode trajectories
        return list(extract(self.strands, reset, fragment))

    def finish(self) -> Any:
        out = [stitch(*fragmets) for fragmets in self.strands.values()]
        # this also decrefs the lists in the strands, which causes
        #  them to decref the tensor strands they contain
        self.strands.clear()

        return out


def empty(
    x: Any,
    dim: tuple[int],
) -> Union[Tensor, ndarray]:
    # handle non-pyt data through numpy
    if isinstance(x, Tensor):
        return x.new_zeros(dim + x.shape)

    # infer the correct basic data type and shape
    x = np.asanyarray(x)
    return np.zeros(dim + x.shape, x.dtype)


def copyto(
    dst: Union[Tensor, ndarray],
    src: Union[Tensor, ndarray],
    *,
    at: int,
) -> None:
    if isinstance(dst, Tensor):
        # `torch.can_cast` is the same as `numpy.can_cast(..., 'same_kind')`
        #  so we compare dtypes directly
        if src.dtype != dst.dtype:
            raise TypeError(f"Cannot cast from `{src.dtype}` to `{dst.dtype}`")

        # `.copy_` makes unsafe dtype casts and host-device moves
        dst[at].copy_(src)

    else:
        # allow onyl strict dtype copies
        np.copyto(dst[at], src, 'no')


class UnorderedLazyBuffer:
    """A lazily initialized buffer for unordered data.
    """
    def __init__(self, capacity: int) -> None:
        self._buffer = None
        self.used = deque([], capacity)
        self.free = set()

    def from_example(self, ex: Any) -> None:

        # pre-allocate the storage
        # XXX consider memmapping large buffers
        #  https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        n_slots = self.used.maxlen
        self._buffer = plyr.apply(empty, ex, dim=(n_slots,))

        # init the free slot stack
        self.free = set(range(n_slots))
        self.used.clear()

    def __getitem__(self, index: int) -> Any:
        """Get the data stored at index.
        """
        return plyr.apply(lambda x: x[index], self._buffer)

    def __setitem__(self, index: int, ex: Any) -> Any:
        """Put the data into the storage at the index.
        """
        # allow negative indexing
        j = (index + self.used.maxlen) if index < 0 else index

        # maintain strict data types
        plyr.apply(copyto, self._buffer, ex, at=j)

    def push(self, ex: Any) -> None:
        """Push the new data into the buffer, optionally evicting the oldest.
        """
        # lazily initialize ourselves
        if self._buffer is None:
            self.from_example(ex)

        # get the next free slot (genuine or evicted)
        j = (self.free or self.used).pop()
        self[j] = ex
        self.used.appendleft(j)

    def pop(self) -> Any:
        """Pop the oldest data from the buffer.
        """
        if self.used:
            j = self.used.pop()
            self.free.add(j)
            return self[j]

        raise IndexError

    def __bool__(self) -> bool:
        """Test whether there is anything in the buffer.
        """
        return bool(self.used)
