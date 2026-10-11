"""Quick-looks: the pictures a run makes as it goes (``gpuwm-cm1 quicklook``).

Generalised from the viz 2.0 quick-look code for any run a CM1 user brings.  CPU only: the worker runs at nice 19
with no GPU visible, and nothing here imports CuPy.

:mod:`.style`         the viz 2.0 look: colours, ramps, type, the frame grammar, the honesty lint
:mod:`.frame`         one published frame, read through the cm1out frame reader
:mod:`.facts`         what every footer says, from the run's own records
:mod:`.reflectivity`  the scheme's own reflectivity or CM1's calc_refl10cm, nothing else
:mod:`.products`      the six quick-looks
:mod:`.swath`         running maxima on the ground (the model's own, or the saved frames')
:mod:`.series`        the time series, with CM1's curve on CM1's stock storm
:mod:`.slices`        horizontal and vertical slices of any cm1out field
:mod:`.skewt`         the skew-T of one model column
:mod:`.render`        one pass over a run's frames, and ``--follow``
"""

from __future__ import annotations

__all__: list[str] = []
