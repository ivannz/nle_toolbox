import plyr
import torch

from collections import defaultdict


def stitch(*chunks, dim=0):
    return plyr.apply(torch.cat, *chunks, _star=False, dim=dim)


def extract(strands, reset, fragment):
    """Generate completed episodes from the given trajectory fragments,
    the corresponding reset mask and incomplete strands.

    Parameters
    ----------
    strands : container of lists
        The incomplete fragmented trajectory traced in each environment.

    reset : boolean torch.Tensor, shape=(T, B)
        A boolean mask, indicating if the associated record in `fragment` is
        marked as the end-of-episode.

    fragment : Nested Container of torch.Tensor, shape=(T, B, ...)
        The new fragment of trajectories with in each environment.

    Yields
    ------
    episode : Nested Container of torch.Tensor, shape=(L, ...)
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
    def __init__(self):
        # the fragmented contiguous trajectory of each incomplete episode
        self.strands = defaultdict(list)

    def extract(self, reset, fragment):
        # the list of completed episode trajectories
        return list(extract(self.strands, reset, fragment))

    def finish(self):
        out = [stitch(*fragmets) for fragmets in self.strands.values()]
        # this also decrefs the lists in the strands, which causes
        #  them to decref the tensor strands they contain
        self.strands.clear()

        return out
