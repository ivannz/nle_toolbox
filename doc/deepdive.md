# A dive int NLE source code

## Architecture
[NetHackRL::getch_method](./nle/win/rl/winrl.cc#422-442) gets called by the main nethack
loop eventually (possibly indirectly via a tty emulator). When called, it yields back
to into python controlled context (stack) with [nle_yield(notdone=TRUE)](./nle/src/nle.c#L326-342).
The latter switches, or `longjump`s, back into [nle_start]() on the very first call,
and then back into [nle_step](./nle/src/nle.c#L423-438) until `done`.
This uses boost's [jump_fcontext]() mechanism instead of callbacks for cleaner c-stack separation.

the nledl.c code is related to dynamically loading the compiled nethack-nle shared object, 
finding symbols for [start](./nle/sys/unix/nledl.c#23), [step](./nle/sys/unix/nledl.c#L32),
and graceful [end](./nle/sys/unix/nledl.c#46). These symbols correspond to the functions in
[nle.c](./nle/src/nle.c).


## Hooking into Nethack proper

The developers of the NLE also provide extensive documentation about the internal
architecture of the project, which, essentially, is a window client (port) for the NetHack
that implements gym api in python. [ARCHITECTURE.md](./nle/doc/nle/ARCHITECTURE.md)

Following the reference tree for `nle_yiled` came up with the following chain:
1. [getch_method](/win/rl/winrl.cc#L423) calls `fill_obs(nle_get_obs());` and
does the context switch, which returns the input character.
2. [rl_nhgetch](/win/rl/winrl.cc#L973) is a plain wrapper around `getch_method`
3. [rl_nh_poskey](/win/rl/winrl.cc#L982) is also a wrapper arounf `rl_nhgetch`, but
it seems that it used to expect a character at certain cursor position.
4. [rl_procs](/win/rl/winrl.cc#L1133) the `window_procs` struct that seems to be a call
table for terminal hooks
5. [window_procs](/include/winprocs.h#L10-83) refers to `rl_nhgetch` and `rl_nh_poskey`
by [`win_nhgetch` and `win_nh_poskey`](/include/winprocs.h#L51-52), respectively.
6. Finally `rl_procs` is added to the list of [win_choices](/src/windows.c#L92-98)
on [lines 135-137](/src/windows.c#L135-137)
7. This is controlled by macro def [RL_GRAPHICS](include/config.h#L53)

Note that no functions that refer to `win_choices` have been modified by the NLE team.
```
(getch_method|rl_nhgetch|rl_nh_poskey|rl_procs)
(winchoices|win_choices_find|choose_windows|addto_windowchain|commit_windowchain)
```
This has been checked by searching through the diff of what has been added to NetHack 3.6
by the NLE:
```bash
# add the original nethack repo
git remote add nethack https://github.com/NetHack/NetHack.git
git fetch nethack

# get the diff with the best common ancestor
git diff -u $(\
    git merge-base remotes/nethack/NetHack-3.6 main\
)..main > nle-2020.diff
```

## the Observation filler

See [NetHackRL::fill_obs](./nle/win/rl/winrl.cc#L259-420)



## Glyphs and terrain passability info

See [rm.h](./nle/include/rm.h) for cmap objects, i.e. room. Contains high level semantics and
meaningful macro defs (see also [floodfillchk_match_accessible](./nle/src/sp_lev.c#L3885-3892)
for special levels). The defs are tightly connected to pathfinding and movement logic of
the [player `u`](./nle/include/you.h#L273-289) and of a [monster](./nle/include/monst.h#L71-173):
[test_move()](./nle/src/hack.c#L709-923) and [mfndpos()](./nle/src/mon.c#L1303-1548), respectively.


