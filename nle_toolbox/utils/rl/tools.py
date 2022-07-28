import plyr
import torch
import numpy as np

from typing import Any, Union
from torch import Tensor
from numpy import ndarray

from collections import defaultdict, deque, namedtuple

Chunk = namedtuple("Chunk", "size,data")


def cat(
    arraylike: Union[Tensor, ndarray],
    dim: int = 0,
) -> Union[Tensor, ndarray]:
    """Concatenate the ndarray or tensor data along the specified dim."""
    if isinstance(arraylike[0], Tensor):
        return torch.cat(arraylike, dim=dim)

    return np.concatenate(arraylike, axis=dim)


def stitch(
    *chunks: Any,
    dim: int = 0,
) -> Any:
    """Stitch the structured fragments.

    Details
    -------
    This procedure accepts a variable number of chunks in through its
    positionals and combines them in to one complete episode (along
    the sequence dim `T`).
    """
    return plyr.apply(cat, *chunks, _star=False, dim=dim)


def extract(
    strands: dict[int, list],
    reset: Union[Tensor, ndarray],
    fragment: Any,
    *,
    copy: bool = True,
) -> tuple[Chunk, ...]:
    """Generate completed episodes from the given batched trajectory fragments,
    the corresponding reset mask and incomplete strands.

    Parameters
    ----------
    strands : container of lists
        The incomplete fragmented trajectory traced in each environment.

    reset : boolean or int array-like, shape=(T, B)
        A boolean or int mask, with NON-ZEROs indicating if the associated
        record in `fragment` is marked as the end-of-episode. Either a numpy
        array or a torch tensor.

    fragment : Nested Container of array-like, shape=(T, B, ...)
        The new fragment of trajectories with in each environment. Can be
        a mix of numpy arrays and torch tensors. It is assumes that the leaf
        data in the container is dimension-synced to `reset`.

    copy : bool, default = False
        Whether to force a copy of the trajectory data when extracting strands.

    Yields
    ------
    chunks : tuple of Chunk
        A tuple of chunks, each containing consecutive fragment elements, which
        together constitute a complete trajectory between AND INCLUDING successive
        non- zeros in `reset`. Chunk report their length in `.size` and content
        in `.data` -- a nested container of array-like of shape `L x B x ...`
        with unitary batch dimension (B=1).

    Details
    -------
    Incomplete trajectories may live in an fragmented state for a long time,
    since the distribution of the spans of full trajectories is likely long
    tailed. At the same time, by default, both numpy and torch produce VIEWS
    and NOT COPIES when slicing and/or doing a single-index access. This may
    lead to a situation, when the memory is congested by a large number of past
    fragments, each referenced ONLY by array views that live in the strands
    of the trajectories of a very few on-going unfinished data sources. See

        https://numpy.org/devdocs/user/basics.indexing.html#advanced-indexing
    """
    # for each independent environment in the batch
    n_seq, n_env = reset.shape
    for j in range(n_env):
        # get the fiber of an env in the batch, keeping the dimension. Indexing
        #  with a single-element list always copies both in numpy and torch,
        #  and slicing from j to j+1 -- always makes a view.
        jj = [j] if copy else slice(j, j + 1)
        # XXX it is better to be explicit, rather than implicit...

        # find all reset brackets [t0, t1)
        t1 = None
        for t in range(n_seq):
            if not reset[t, j]:  # XXX ok with ternary, as ZERO is False
                continue

            # finish with the [t0, t1+1) slice, including the `t1`-th element,
            #  which could contain data, relevant to the end of the trajectory
            t1, t0 = t, t1
            tail = plyr.apply(lambda x: x[t0 : t1 + 1, jj], fragment)
            strands[j].append(Chunk(t1 + 1 - (t0 or 0), tail))

            # combine the fragments together and drop the incomplete strand
            yield tuple(strands[j])
            strands[j].clear()

        # commit the residual piece [t1, +oo) to the strand
        chunk = plyr.apply(lambda x: x[t1:, jj], fragment)
        strands[j].append(Chunk(n_seq - (t1 or 0), chunk))


def add_leading(x: Any, *, n_leading: int = 0) -> Union[Tensor, ndarray]:
    """View the input as an array with extra unitary leading dimensions.

    Parameters
    ----------
    x : arraylike or Tensor
        A tensor, ndarray or an array-like sequence. Any non-array input is
        converted to a numpy array.

    Returns
    -------
    result : ndarray or Tensor
        An array or tensor with extra unitary leading dimensions. Copies are
        avoided where possible.
    """
    # let numpy handle everything that is not a tensor
    if not isinstance(x, Tensor):
        x = np.asanyarray(x)

    return x.reshape(n_leading * (1,) + x.shape) if n_leading > 0 else x


def ensure2d(
    reset: Union[Tensor, ndarray],
    fragment: Any,
) -> tuple[Union[Tensor, ndarray], Any]:
    """Add as many LEADING dims to the fragment as were missing in `reset`.

    Details
    -------
    `numpy.atleast2d(*arys)` casts torch's tensors to npy's ndarrays, while
    `torch.atleast_2d` refuses to accept non-array inputs. Besides, we wan to
    expand the dims only when the reference `reset` array is missing some.
    """
    # convert non-array reset to ndarray, unless a tensor or already an ndarray
    if not isinstance(reset, Tensor):
        reset = np.asanyarray(reset)

    # silently allow higher than 2d arrays
    if reset.ndim >= 2:
        return reset, fragment

    # add the missing dims to the reset array
    n_missing = 2 - reset.ndim
    reset = reset.reshape(n_missing * (1,) + reset.shape)

    # add extra unitary leading dims to the leaves of the structured fragment
    return reset, plyr.apply(add_leading, fragment, n_leading=n_missing)


class EpisodeExtractor:
    """A simple object to track and stitch fragmented trajectories."""

    def __init__(self) -> None:
        # the fragmented contiguous trajectory of each incomplete episode
        self.strands = defaultdict(list)

    def extract(
        self,
        reset: Union[Tensor, ndarray],
        fragment: Any,
    ) -> list[Any]:
        # make sure that we're feeding `T x B` data to the extractor
        reset, fragment = ensure2d(reset, fragment)

        # the list of completed episode trajectories
        episodes = []
        for out in extract(self.strands, reset, fragment):
            _, chunks = zip(*out)
            episodes.append(stitch(*chunks))

        return episodes

    def finish(self) -> Any:
        # stitch together the remaining chunks of the unfinished episodes
        episodes = []
        for out in self.strands.values():
            _, chunks = zip(*out)
            episodes.append(stitch(*chunks))

        # `.clear` also decrefs the lists in the strands, which causes
        #  them to eventually decref the tensor strands they contain.
        self.strands.clear()
        return episodes


def empty(
    x: Any,
    dim: tuple[int],
) -> Union[Tensor, ndarray]:
    # non-pyt data is handled by numpy
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
        # allow only strict dtype copies
        np.copyto(dst[at], src, "no")


class UnorderedLazyBuffer:
    """A lazily initialized buffer for unordered data."""

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
        """Get the data stored at index."""
        return plyr.apply(lambda x: x[index], self._buffer)

    def __setitem__(self, index: int, ex: Any) -> Any:
        """Put the data into the storage at the index."""
        # allow negative indexing
        j = (index + self.used.maxlen) if index < 0 else index

        # maintain strict data types
        plyr.apply(copyto, self._buffer, ex, at=j)

    def push(self, ex: Any) -> None:
        """Push the new data into the buffer, optionally evicting the oldest."""
        # lazily initialize ourselves
        if self._buffer is None:
            self.from_example(ex)

        # get the next free slot (genuine or evicted)
        j = (self.free or self.used).pop()
        try:
            self[j] = ex

        except TypeError:
            # mark the fetched slot as free on any error
            self.free.add(j)
            raise

        else:
            # otherwise use it up
            self.used.appendleft(j)

    def pop(self) -> Any:
        """Pop the oldest data from the buffer."""
        if self.used:
            j = self.used.pop()
            self.free.add(j)
            return self[j]

        raise IndexError

    def __bool__(self) -> bool:
        """Test whether there is anything in the buffer."""
        return bool(self.used)


def fetch(x, t0, t1):
    """Copy a padded slice [t0, t1) from the arraylike `x`."""

    # if the left endpoint is negative, then pad with the first element
    return cat((x[:1],) * max(-t0, 0) + (x[max(t0, 0) : t1],))


class EpisodeBuffer:
    """Keeping at least the specified number of transitions."""

    def __init__(self, *, seed: Any = None) -> None:
        self.n_transitions = 0
        self.storage = deque([])
        self.strands = defaultdict(list)
        self.random_ = np.random.default_rng(seed=seed)

    def __len__(self):
        return self.n_transitions

    @property
    def n_episodes(self):
        """Get the number of complete episodes."""
        return len(self.storage)

    @property
    def n_pending(self):
        """Get the number of pending transitions in unfinished episodes."""
        n_pending = 0
        for chunks in self.strands.values():
            sizes, _ = zip(*chunks)
            n_pending += sum(sizes)

        return n_pending

    def shrink(self, n_target: int = None) -> None:
        """Shrink the buffer to the specified number of transitions."""
        if n_target is None:
            n_target = self.n_transitions - 1

        # evict the oldest episodes until we hit the population target
        while self.n_transitions > n_target:
            size, _ = self.storage.popleft()
            self.n_transitions -= size - 1

        return len(self)

    def push(
        self,
        reset: Union[Tensor, ndarray],
        fragment: Any,
        *,
        weight: Union[Tensor, ndarray] = None,
    ) -> None:
        # make sure that we're feeding `T x B` data to the extractor
        reset, fragment = ensure2d(reset, fragment)
        # XXX we could also associate a weight series with each fragment,
        #  every value of which weighs the observation, e.g. td-error.

        # get the complete episodes (strands store slice view into fragements)
        for out in extract(self.strands, reset, fragment):
            # combine the chunks into an episode
            sizes, chunks = zip(*out)
            episode = Chunk(sum(sizes), stitch(*chunks))

            # by design valid episodes contain at least two steps
            if episode.size < 2:
                continue

            assert bool(episode.data.fin[0]), "This should never happen."

            # save the episode and update the transition population size
            self.storage.append(episode)
            self.n_transitions += episode.size - 1

    def sample(
        self,
        n_transitions: int,
        n_batch: int,
        *,
        # for the recurrent runtime state wram-up (>= 0)
        n_burnin: int = 0,
        # the numbr of elements for frame stacking (>= 1)
        n_window: int = 1,
    ) -> tuple[Any, Any]:
        """Randomly sample continuous trajectory fragments with burnin."""
        burnin = None

        # the number of elements in the batch proper (not transitions!!!) (>= 1)
        n_length = n_transitions + 1

        # the total number of elements to request from each trajectory
        n_total = n_length + n_burnin + max(n_window - 1, 0)

        # make an O(n) shallow copy for faster random access
        episodes = tuple(self.storage)

        # select `n_batch` random episode indices
        epix = self.random_.integers(len(episodes), size=n_batch)
        sizes, batch = zip(*[episodes[j] for j in epix])
        sizes = np.array(sizes)

        # for each picked trajectory draw an index of the right endpoint
        # for the slice [t0, t1) with $t1 \sim \{\min\{L, T_j\} .. T_j\}$
        left = np.minimum(sizes, n_length)
        stix = self.random_.integers(left, sizes, endpoint=True)

        # pad the selected episodes
        padded = [
            plyr.apply(fetch, ep, t0=t0, t1=t1)
            for ep, t0, t1 in zip(batch, stix - n_total, stix)
        ]

        # collate into a batch
        padded = plyr.apply(cat, *padded, _star=False, dim=1)

        # split the batch into the proper and brunin segments
        batch = plyr.apply(lambda x: x[1 - n_window - n_length :], padded)
        if n_burnin > 0:
            burnin = plyr.apply(lambda x: x[:-n_length], padded)

        return batch, burnin
