"""Run a CM1 case with gpuwm-cm1: price it, fit it, start it, watch it.

The run directory (``gm1.cm1-run/v1``) is the contract between the command
line, a batch job and the local GUI: a run
started from a terminal, a Slurm job or the GUI looks the same to each of
them.

This package never imports CuPy or opens a device, and importing any of its
modules imports neither CuPy nor numpy; every module in it can be called from
the GUI server, a login node or a test without a GPU.

Pricing and fitting:

:mod:`.gpuinfo`
    the GPUs ``nvidia-smi`` reports, with the memory actually free, and the
    host's RAM and disk;
:mod:`.rates`
    the measured step rates (``records/GM1-GPU-RATES.json``) and the ones this
    machine has measured itself (``~/.cache/gpuwm-cm1/rates.json``);
:mod:`.estimate`
    ``price(deck, gpu)``: GPU memory, locked host memory, disk and wall time
    for one deck on one GPU, the storm-track check, and the other checks
    ``gpuwm-cm1 estimate`` prints;
:mod:`.fit`
    ``fit_preset(preset, gpu, budget)``: the guided mode's auto-fit, which
    shrinks, coarsens or shortens a preset until it fits a GPU and a time
    budget.

Running:

:mod:`.rundir`
    the layout, ``run.json`` and each segment's record, and which frames and
    restarts form the run;
:mod:`.events`
    reading a segment's ``gm1.run-events/v1`` log as it grows;
:mod:`.status`
    what a run is doing, with host-aware crash detection;
:mod:`.control`
    stop, pause, terminate and force stop;
:mod:`.launch`
    ``gpuwm-cm1 run`` and ``resume``: the per-GPU queue, the host admission
    and the supervisor that owns one segment's integration;
:mod:`.jobscript`
    Slurm and PBS scripts with compute-node sidecars.
"""
