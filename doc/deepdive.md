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


The developers of the NLE provide also extensive docs about the internal architecture of
the project, which, essentially, is a wrapper around the original nethack itself.
[See arch](./nle/doc/nle/ARCHITECTURE.md).


### Observation filler

See [NetHackRL::fill_obs](./nle/win/rl/winrl.cc#L259-420)



## Glyphs and terrain passability info

See [rm.h](./nle/include/rm.h) for cmap objects, i.e. room. Contains high level semantics and
meaningful macro defs (see also [floodfillchk_match_accessible](./nle/src/sp_lev.c#L3885-3892)
for special levels). The defs are tightly connected to pathfinding and movement logic of
the [player `u`](./nle/include/you.h#L273-289) and of a [monster](./nle/include/monst.h#L71-173):
[test_move()](./nle/src/hack.c#L709-923) and [mfndpos()](./nle/src/mon.c#L1303-1548), respectively.


