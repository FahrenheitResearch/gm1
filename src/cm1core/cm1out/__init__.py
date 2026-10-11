"""CM1-format output from GM1 history frames (``gpuwm-cm1 cm1out``).

:mod:`.frames`    published frames and run layouts (never ``.incomplete/``);
:mod:`.settings`  the CM1 settings (namelist, else deck) and the frame's grid;
:mod:`.names`     CM1 r21.1 writeout.F's names, descriptions, units and order;
:mod:`.fields`    each selected CM1 variable read from a frame, or why not;
:mod:`.netcdf`    writeout_nc.F's history-file layout and global attributes;
:mod:`.grads`     writeout.F's GrADS records and descriptors (``output_format = 1``);
:mod:`.stats`     statpack.F's statistics from frames, and ``cm1out_stats.nc``;
:mod:`.pdata`     parcel.F's ``cm1out_pdata`` from the run's parcel track;
:mod:`.reflectivity` morrison.F's calc_refl10cm: Morrison ``dbz`` and ``cref`` on the CPU;
:mod:`.convert`   the conversion pass and ``--follow``.

Nothing here imports CuPy or touches a GPU.
"""

from __future__ import annotations

from .convert import Cm1outError, Converter, Summary, convert

__all__ = ["Cm1outError", "Converter", "Summary", "convert"]
