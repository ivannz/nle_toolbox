"""Low-level seeding of Nethack env for action-deterministic replay."""

import numpy
from numpy.random import default_rng

from nle import nethack
from nle.nethack import Nethack as pyNethack
from nle._pynethack import Nethack as cNethack


def is_seedable():
    """Check if the low-level NEthack env supports seeding."""
    return getattr(nethack, 'NLE_ALLOW_SEEDING', False) \
        and getattr(cNethack, 'set_initial_seeds', False)


def pyroot(env):
    """Locate the root cNethack c-implementation in the chain of `.env`-s"""
    if isinstance(env, cNethack):
        return env

    # get to the bottom of this
    root = env
    try:
        while not isinstance(env, pyNethack):
            env = env.env

        env = env._pynethack

    except AttributeError:
        pass

    else:
        if isinstance(env, cNethack):
            return env

    raise RuntimeError(
        f"`{root}` does not appear to be a proper Nethack env."
    ) from None


def generate(seed=None, *, maxint=numpy.iinfo(numpy.uint).max):
    """Produce a deterministic pair of seeds for the root cNethack env."""

    # have to generate a seed from the sequence manually, since
    #  nle passes it directly to `cNethack.set_initial_seeds`
    rng = default_rng(seed)

    # `set_initial_seeds` uses two unsigned-longs (`numpy.uint` dtype)
    core, disp = map(int, rng.integers(maxint, size=2, dtype=numpy.uint))
    return core, disp


def set_seed(env, *, seed=None):
    """Locate the lowest-level pyNethack environment and seed it.

    Details
    -------
    Some facts about `cNethack` gathered from the apparent behaviour and a dive
    into its cpp code in `win/rl/pynethack.cc`

        * [set_initial_seeds](win/rl/pynethack.cc#L193) uses the provided seeds
        to generate the entire game and determine the future gameplay chaos
        until [the next](win/rl/pynethack.cc#L260) call to
        [reset(FILE *)](win/rl/pynethack.cc#L249).

        * [set_seeds](win/rl/pynethack.cc#L206) affects the _current_ state of
        the internal generator by calling [nle_set_seed](src/nle.c#L469)

        * [get_seeds](win/rl/pynethack.cc#L218) returns the seeds of
        the [current](win/rl/pynethack.cc#L226) chaos generator using
        [nle_get_seed](src/nle.c#L483), ONLY after the env has been reset
        at [least once](win/rl/pynethack.cc#L221).

    Procedures in `src/nle.c` are not even compiled if `NLE_ALLOW_SEEDING`
    (`nle.nethack.NLE_ALLOW_SEEDING`) macro def is not set. The procs in
    `win/rl/pynethack.cc` raise a std-lib's runtime error instead. Line
    numbers are valid as of commit `ce577cb1c` of nle's official github repo.

    Warning
    -------
    Although this makes the observation sequences deterministically dependent
    on the actions, taken in the env, it is possible that certain non-critical
    fields in the observation dict deviate between even seeded runs if the same
    instance of the Nethack env is reused. Namely, the fields `tty_chars`,
    `tty_colors` and `tty_cursor` might deviate due to the cursor location
    in the emulated tty terminal. Other fields, most importantly, `glyphs`,
    `message` and `inv_*` are identical.
    """
    core, disp = generate(seed)

    root = pyroot(env)
    try:
        # set the initial seeds, disabling the TAS protection
        root.set_initial_seeds(core, disp, False)

    except RuntimeError:
        # Nethack raises, if it has not been compiled with seeding
        # support. We re-raise.
        raise

    return core, disp
