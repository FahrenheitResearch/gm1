"""Films of the quick-looks: ``gpuwm-cm1 quicklook RUNDIR --film LOOK``.

Four kinds of film, all written under ``RUNDIR/quicklook/films/LOOK/``:

* any per-frame look (the wind swath building up, reflectivity, clouds seen from above, ...): its own saved
  pictures, one per saved time, in order; nothing is drawn again;
* ``swirl``: the spin look's streaks animated.  Each saved time, plus ``between`` pictures between two saved
  times, in which the lowest-level wind and spin are blended linearly in time (said on every picture), and a
  noise texture is carried along the wind and drawn as streaks (line-integral convolution);
* ``parcels``: the run's air parcels at every saved parcel time (:mod:`.parcels`);
* ``storm3d``: one turn around the storm in 3-D, cut open, at one saved time (:mod:`.storm3d`).

Always written, with no extra dependency: the pictures (``frames/``, or the look's own), and an animated PNG
(``film.png``, APNG: every browser plays it).  An MP4 (H.264, yuv420p) is written when ``imageio_ffmpeg`` is
installed (``pip install "gpuwm-cm1[film]"``): it runs the separately licensed ffmpeg program imageio-ffmpeg
provides, as its own process; nothing of it is in this package.  A GIF is written on request, or in the MP4's
place when imageio-ffmpeg is not installed.

``films/LOOK/progress.json`` says how far a film is while it is made (the GUI's progress line), and
``films/index.json`` lists the films made.  SIGTERM or SIGINT stops a film between pictures; what was made is
kept and the progress says it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time as _time
from typing import Any, Callable, Iterable, Sequence
import zlib

import numpy as np

from . import products as P
from . import style as S

FILMS = "films"
INDEX = "index.json"
INDEX_SCHEMA = "gm1.quicklook-films/v1"
PROGRESS_SCHEMA = "gm1.quicklook-film-progress/v1"
FORMATS = ("apng", "mp4", "gif")
DERIVED = ("swirl", "parcels", "storm3d")
SWIRL_SIZE = (1280, 720)
APNG_SIZE = (1280, 720)
GIF_SIZE = (960, 540)
#: Colours a look's GIF keeps exactly, whatever area they cover (look id: "module:name" in this package, a function
#: giving RGB 0..255 triples).  The median cut gives each GIF picture's 255 colours to its largest areas, so a few
#: pixels of the infrared look's coldest tops (purple, pink) would take the nearest large colour, a grey that the
#: picture's colours say is warmer than -30 C.  The other looks' GIFs are made as before.
GIF_KEEP = {"infrared": "infrared:gif_keep"}
DEFAULT_FPS = {"swirl": 18, "parcels": 12, "storm3d": 12}
DEFAULT_BETWEEN = 8
TITLES = {"swirl": "Swirl near the ground", "parcels": "Air rising into the storm",
          "storm3d": "The storm in 3-D, one turn around it"}


class FilmError(ValueError):
    """A film that cannot be made, in plain words (and the fix)."""


class FilmKept(FilmError):
    """A film made from the run's data that is not made again over the one it has: made now, it would show less
    (saved frames deleted since).  Nothing is changed, its progress included."""


def remake_refused(job: "FilmJob") -> str | None:
    """Why a film of the swirl, the parcels or the storm in 3-D is not made again over the one the run has, or None.

    These are made from the saved frames (the parcels from the track kept beside them), not from pictures: while
    any drawn saved time has lost its frame (:func:`.render.lost_times`), a swirl or parcels film made again would
    show less than the one it replaces, and the 3-D film it has, if its own saved time is one of those, could not be
    made again at all."""

    from .frame import frames_of, same_frame
    from .ncread import discover
    from .render import drawn_rows, lost_times

    made = (_read_films(job.out).get(job.look) or {})
    if not made and not (job.folder / "film.png").is_file():
        return None
    try:
        layout = discover(job.target)
    except ValueError:
        return None          # the film's own reading says why it cannot be made
    lineage = frames_of(layout)
    rows = drawn_rows(job.out)
    lost = lost_times(layout, lineage, rows)
    if not lost:
        return None
    if job.look == "storm3d":
        frame = ((made.get("facts") or {}) if isinstance(made.get("facts"), dict) else {}).get("frame")
        if frame and any(same_frame(frame, key, None, segment) for _t, key, _p, segment in lineage):
            return None      # its own saved time is still there: it can be made again
        return ("The 3-D film this run has was made at a saved time whose frame was deleted, so it could not be made "
                "again, and a new one would replace it. Nothing was changed; the film it has is kept.")
    source = "the parcel track kept beside the saved frames" if job.look == "parcels" else "the saved frames"
    return (f"{len(lost)} of this run's {len(rows)} drawn saved times have lost their saved frames, and this film is "
            f"made from {source}: made again, it would show less than the one it has. Nothing was changed; the "
            "film it has is kept.")


def _read_films(out: Path) -> dict[str, Any]:
    try:
        index = json.loads((out / FILMS / INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    films = index.get("films") if isinstance(index, dict) else None
    return films if isinstance(films, dict) else {}


def mp4_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    return True


def _utc() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- encoders
def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    out, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        out.append((kind, data[pos + 8:pos + 8 + length]))
        pos += 12 + length
    return out


def _encode_png(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def write_apng(images: Iterable[Any], count: int, out: Path, fps: float) -> Path:
    """An animated PNG, streamed one picture at a time (every picture a full frame, RGB, looping).

    The container is written by hand (PNG chunks, the APNG ``acTL``/``fcTL``/``fdAT`` chunks), so no picture is
    held in memory after it is written: a 200-picture film needs one picture's memory."""

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    delay = max(1, int(round(1000.0 / fps)))
    sequence = 0
    written = 0
    with tmp.open("wb") as stream:
        for number, image in enumerate(images):
            chunks = _png_chunks(_encode_png(image.convert("RGB")))
            header = next(data for kind, data in chunks if kind == b"IHDR")
            width, height = struct.unpack(">II", header[:8])
            if number == 0:
                stream.write(b"\x89PNG\r\n\x1a\n")
                stream.write(_chunk(b"IHDR", header))
                stream.write(_chunk(b"acTL", struct.pack(">II", count, 0)))
                size = (width, height)
            elif (width, height) != size:
                raise FilmError("every picture of a film must be the same size")
            stream.write(_chunk(b"fcTL", struct.pack(">IIIIIHHBB", sequence, width, height, 0, 0, delay, 1000, 0, 0)))
            sequence += 1
            for kind, data in chunks:
                if kind != b"IDAT":
                    continue
                if number == 0:
                    stream.write(_chunk(b"IDAT", data))
                else:
                    stream.write(_chunk(b"fdAT", struct.pack(">I", sequence) + data))
                    sequence += 1
            written += 1
        stream.write(_chunk(b"IEND", b""))
    if written != count:
        # the header promised ``count`` pictures: rewrite acTL with what was written (a stopped film)
        data = bytearray(tmp.read_bytes())
        at = data.find(b"acTL")
        data[at + 4:at + 8] = struct.pack(">I", written)
        data[at + 12:at + 16] = struct.pack(">I", zlib.crc32(bytes(data[at:at + 12])) & 0xFFFFFFFF)
        tmp.write_bytes(bytes(data))
    os.replace(tmp, out)
    return out


def read_apng_frames(path: Path) -> tuple[int, int]:
    """(pictures, plays) an APNG declares in its ``acTL`` (for tests and the index)."""

    data = path.read_bytes()
    for kind, body in _png_chunks(data):
        if kind == b"acTL":
            return struct.unpack(">II", body)
    return 1, 1


def gif_keep(look: str) -> list[tuple[int, int, int]]:
    """The colours ``look``'s GIF keeps exactly (:data:`GIF_KEEP`), or none."""

    import importlib

    where = GIF_KEEP.get(look)
    if not where:
        return []
    module, name = where.split(":")
    colours = getattr(importlib.import_module(f".{module}", __package__), name)()
    return [(int(r), int(g), int(b)) for r, g, b in colours]


def _gif_picture(image: Any, keep: Sequence[tuple[int, int, int]], palette: Any = None) -> Any:
    """One GIF picture: the median cut's 255 colours, or with ``keep`` its fewer colours and those, each pixel the
    nearest of them; given ``palette`` (a "P" image), that one palette."""

    from PIL import Image

    rgb = image.convert("RGB").resize(GIF_SIZE, Image.LANCZOS)
    if palette is not None:
        return rgb.quantize(palette=palette, dither=Image.NONE)
    if not keep:
        return rgb.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)
    cut = rgb.quantize(colors=255 - len(keep), method=Image.MEDIANCUT, dither=Image.NONE)
    palette = list(cut.getpalette() or [])[:3 * (255 - len(keep))] + [v for colour in keep for v in colour]
    palette += palette[:3] * ((768 - len(palette)) // 3)             # unused entries repeat the first colour
    holder = Image.new("P", (1, 1))
    holder.putpalette(palette)
    return rgb.quantize(palette=holder, dither=Image.NONE)


def write_gif(images: Iterable[Any], out: Path, fps: float, palette: Any = None, *,
              keep: Sequence[tuple[int, int, int]] = ()) -> Path:
    """A GIF at :data:`GIF_SIZE`: each picture cut to its own 255 colours (keeping ``keep`` exactly), or, given
    ``palette`` (a "P" image), every picture to that one palette (the storm in 3-D, whose legend colours must come
    out exactly)."""

    frames = [_gif_picture(image, keep, palette) for image in images]
    if not frames:
        raise FilmError("no pictures to make a GIF from")
    tmp = out.with_name(f".{out.name}.partial")
    frames[0].save(tmp, format="GIF", save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)),
                   loop=0, optimize=False, disposal=1)
    os.replace(tmp, out)
    return out


def _mp4_fix() -> str:
    """How to add the film add-on for the way GM1 was installed (the installer brings it; pip adds the extra)."""

    try:
        from ..cm1run.system import film_fix

        return film_fix()
    except Exception:  # noqa: BLE001 - the words never stop a film
        return 'pip install "gpuwm-cm1[film]"'


def write_mp4(images: Iterable[Any], size: tuple[int, int], out: Path, fps: float) -> Path:
    """H.264 in MP4 through imageio-ffmpeg's own ffmpeg program (the optional ``film`` extra)."""

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise FilmError(f"An MP4 needs the optional film add-on: {_mp4_fix()}. "
                        "The animated PNG and the pictures are written without it.") from error
    tmp = out.with_name(f".{out.name}.partial.mp4")
    writer = imageio_ffmpeg.write_frames(os.fspath(tmp), size, fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                         pix_fmt_out="yuv420p", quality=8, macro_block_size=8,
                                         ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
    writer.send(None)
    for image in images:
        rgb = image.convert("RGB")
        if rgb.size != size:
            from PIL import Image

            rgb = rgb.resize(size, Image.LANCZOS)
        writer.send(np.asarray(rgb, np.uint8).tobytes())
    writer.close()
    os.replace(tmp, out)
    return out


# ----------------------------------------------------------------------------------------------- pictures
def save_picture(ov: S.Overlay, base: np.ndarray, path: Path) -> Path:
    """Lint, composite and write one film picture (no thumbnail)."""

    from PIL import Image

    problems = ov.check()
    if problems:
        raise P.LintError(f"{path.name}: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    Image.fromarray(ov.composite(base)).save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, path)
    return path


@dataclass
class Swirl:
    """The swirl film's state: the texture carried along the wind from one picture to the next."""

    box: P.Box
    noise: np.ndarray
    fresh: np.ndarray
    rng: float
    inject: float = 0.10
    lic_px: int = 12


def _sampled(qf: Any, box: P.Box) -> dict[str, np.ndarray]:
    u, v = qf.uv_level(0)
    zeta = qf.zeta_level(0)[0]
    return {"u": P.sample(u, box, qf.dx, qf.dy), "v": P.sample(v, box, qf.dx, qf.dy),
            "zeta": P.sample(zeta, box, qf.dx, qf.dy), "t": qf.t, "domainlocx": qf.domainlocx,
            "domainlocy": qf.domainlocy, "umove": qf.umove, "vmove": qf.vmove}


def _advect(img: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """``img`` carried ``du`` pixels right and ``dv`` pixels down (semi-Lagrangian: sampled upstream)."""

    h, w = img.shape
    rows = np.arange(h, dtype=np.float32)[:, None] - dv
    cols = np.arange(w, dtype=np.float32)[None, :] - du
    return P.bilinear(img, rows, cols)


def swirl_picture(state: Swirl, fields: dict[str, Any], dt_s: float, facts: Any, extent: tuple[float, float],
                  dx: float, blended: bool, interval: float | None, out: Path, height_m: float) -> dict[str, Any]:
    box = state.box
    u, v, zeta = fields["u"], fields["v"], fields["zeta"]
    # carry the texture with the air across the picture (the domain's own frame: the picture moves with it)
    if dt_s > 0:
        state.noise = _advect(state.noise, u * dt_s / box.mpp, -v * dt_s / box.mpp)
        state.noise = (1.0 - state.inject) * state.noise + state.inject * state.fresh
    sp = np.hypot(u, v) + 1e-6
    L = max(4, int(round(state.lic_px * box.h / 800)))
    tx = P.lic(state.noise, u / sp, -v / sp, L)
    tx = (tx - float(tx.mean())) / max(float(tx.std()), 1e-6) * 0.5
    base = S.RAMPS["zeta"].colours(zeta, -state.rng, state.rng)
    a = (np.clip(0.55 + 0.45 * tx, 0.0, 2.0) * np.clip(sp / 25.0, 0.25, 1.4) * 0.22)[..., None]
    rgb = base * (1.0 - np.clip(a, 0, 0.6)) + np.array(S.hex_rgb("silk"), np.float32) * 1.6 * a
    img, spacing = P.ground_ticks(S.to8(rgb), box, fields["domainlocx"], fields["domainlocy"])
    W, H = SWIRL_SIZE
    cv = np.tile(np.array(S.rgb8("void"), np.uint8), (H, W, 1))
    P.paste(cv, img, box)
    cv = P.outline(cv, box)
    ov = S.Overlay((W, H))
    lx, ly = extent
    moving = fields["umove"] != 0.0 or fields["vmove"] != 0.0
    frame_words = "relative to the moving domain" if moving else "ground-relative"
    ov.title("Swirl near the ground", S.data_line(facts.name, f"{S.fmt_grid(dx)} grid",
                                                  f"spin at {S.fmt_height(height_m)}",
                                                  f"streaks: the wind {frame_words}"))
    ov.clock((getattr(facts, "clock", None) or S.fmt_clock)(fields["t"]))
    every = f" (every {interval:g} s)" if interval else ""
    ov.footer(facts.footer(t=fields["t"], level=f"lowest level {S.fmt_height(height_m)}",
                           wind="domain-relative" if moving else "ground-relative",
                           notes=S.data_line(("blended between saved times" + every) if blended
                                             else "a saved time", "streaks move with the air",
                                             f"spin ±{state.rng:g} {S.PER_S} fixed "
                                             f"{P.spin_range_words(dx, height_m, hurricane=_hurricane(facts))}")))
    ov.mark()
    ov.ramp_bar("zeta", (box.x + box.w - ov.U(360), box.y + box.h + ov.U(16), box.x + box.w,
                         box.y + box.h + ov.U(16) + max(3, ov.U(7))), vmin=-state.rng, vmax=state.rng,
                ticks=[(-state.rng, P.spin_tick(-state.rng, S.fmt_num(-state.rng, 2))), (0.0, "0"),
                       (state.rng, f"{P.spin_tick(state.rng, f'{state.rng:.2f}')} {S.PER_S}")],
                title="clockwise | counter-clockwise")
    save_picture(ov, cv, out)
    return {"time_s": float(fields["t"]), "blended": blended}


def _hurricane(facts: Any) -> bool:
    """Whether the film is of a hurricane run (its swirl range is :func:`.products.swirl_range`'s hurricane one)."""

    return getattr(facts, "storm_kind", "") == "hurricane"


def _blend(a: dict[str, Any], b: dict[str, Any], f: float) -> dict[str, Any]:
    out = {}
    for key, value in a.items():
        other = b[key]
        out[key] = (1.0 - f) * value + f * other if isinstance(value, (np.ndarray, float, int)) else value
    return out


# ----------------------------------------------------------------------------------------------- the door
@dataclass
class FilmJob:
    target: Path
    look: str
    formats: tuple[str, ...]
    fps: float
    between: int
    out: Path
    label: str | None = None
    log: Callable[[str], None] = lambda _m: None
    should_stop: Callable[[], bool] = lambda: False
    started: float = field(default_factory=_time.perf_counter)
    #: storm3d only: the saved time (default the latest), the pictures in one turn, their size, the first bearing
    time_s: float | None = None
    pictures: int | None = None
    size: tuple[int, int] | None = None
    bearing: float | None = None

    @property
    def folder(self) -> Path:
        return self.out / FILMS / self.look

    def progress(self, done: int, total: int, state: str = "running", message: str = "") -> None:
        doc = {"schema": PROGRESS_SCHEMA, "look": self.look, "state": state, "done": done, "total": total,
               "pid": os.getpid(), "updated_utc": _utc(), "seconds": round(_time.perf_counter() - self.started, 1),
               "message": message}
        _write_json(self.folder / "progress.json", doc)


def _read_index(out: Path) -> dict[str, Any]:
    try:
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


def _index_frames(out: Path, look: str) -> list[dict[str, Any]]:
    for product in _read_index(out).get("products") or []:
        if product.get("id") == look:
            return [f for f in product.get("frames") or [] if isinstance(f.get("file"), str)]
    return []


def _index_not_drawn(out: Path, look: str) -> str | None:
    """The reason the index gives for not drawing ``look`` at its latest saved time, or None."""

    try:
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    products = (index.get("products") or []) if isinstance(index, dict) else []
    for product in products:
        if isinstance(product, dict) and product.get("id") == look:
            rows = [r for r in product.get("not_drawn") or [] if isinstance(r, dict) and r.get("reason")]
            latest = max(rows, default=None, key=lambda r: max((t for t in r.get("times_s") or []
                                                               if isinstance(t, (int, float))), default=-1.0))
            return None if latest is None else str(latest["reason"])
    return None


def why_no_pictures(out: Path, look: str, target: Path | str | None = None) -> str:
    """Why a per-frame look has no pictures to film, in plain words: a hurricane's own look in any other run, the
    reason it was not drawn (a look made for thunderstorms, in a hurricane run; a grid it is not drawn for), a
    finished run whose pictures were drawn before the look was offered (with the command that draws it, given
    ``target``), or that none has landed yet."""

    index = _read_index(out)
    run = index.get("run") if isinstance(index.get("run"), dict) else {}
    if look in P.HURRICANE_ONLY and (run or index.get("frames")) and run.get("storm_kind") != "hurricane":
        return f"The {look} look is drawn for hurricane runs only, and this run is not one."
    reason = _index_not_drawn(out, look)
    if reason:
        return reason
    if target is not None and index.get("run_ended") and index.get("frames"):
        return (f"The {look} look was not drawn for this run's pictures (they were drawn before it was offered, or "
                f"without it). Draw it first: gpuwm-cm1 quicklook {target} (in the app: Results, Draw the new "
                "looks).")
    return f"The {look} look has no pictures yet: they are drawn as the frames land."


def per_frame_looks() -> tuple[str, ...]:
    return tuple(pid for pid, _t in P.PRODUCTS)


def film_looks() -> tuple[str, ...]:
    return (*per_frame_looks(), *DERIVED)


def make(job: FilmJob) -> dict[str, Any]:
    """Make one film; returns the film's record (also written to ``films/index.json``).

    A film made from the run's data (:data:`DERIVED`) draws its pictures again into ``frames/``: the ones it had
    are set aside first and put back when the new film is not made (refused, failed or stopped by an error), so a
    film is never left without its pictures."""

    frames_dir = job.folder / "frames"
    before = job.folder / ".frames-before"
    if job.look in DERIVED and before.is_dir():
        # A remake stopped part way (killed) left the pictures of the film on record set aside: they go back first,
        # over what that remake had drawn, and are set aside again below.
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.replace(before, frames_dir)
    set_aside = job.look in DERIVED and frames_dir.is_dir()
    if set_aside:
        os.replace(frames_dir, before)
    try:
        record = _make(job)
    except BaseException:
        if set_aside:
            shutil.rmtree(frames_dir, ignore_errors=True)
            os.replace(before, frames_dir)
        raise
    if set_aside:
        shutil.rmtree(before, ignore_errors=True)
    return record


def _make(job: FilmJob) -> dict[str, Any]:
    from PIL import Image

    look = job.look
    if look not in film_looks():
        raise FilmError(f"There is no film of {look!r}. Films: {', '.join(film_looks())}.")
    unknown = [f for f in job.formats if f not in FORMATS]
    if unknown:
        raise FilmError(f"No such film format: {', '.join(unknown)} (they are {', '.join(FORMATS)}).")
    if "mp4" in job.formats and not mp4_available():
        raise FilmError(f"An MP4 needs the optional film add-on: {_mp4_fix()}. "
                        "Leave out --format mp4 for the animated PNG and the pictures, which need nothing more.")
    if look in DERIVED:
        refused = remake_refused(job)
        if refused:
            raise FilmKept(refused)
    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    job.progress(0, 0, message="starting")
    blended_words = ""
    try:
        if look in DERIVED:
            pictures, size, facts = _derived(job)
            if look == "swirl":
                blended_words = "blended between saved times"
        else:
            frames = _index_frames(job.out, look)
            if not frames:
                # the look's own reason when the index gives one, not "its pictures are still to come"
                reason = _index_not_drawn(job.out, look)
                raise FilmError(f"The {look} look has no pictures to film. {reason}" if reason
                                else why_no_pictures(job.out, look, job.target))
            pictures = [job.out / f["file"] for f in frames]
            size = (P.W, P.H)
            facts = {"times_s": [f.get("time_s") for f in frames]}
            job.progress(len(pictures), len(pictures), message="the look's own pictures")
    except FilmError:
        raise
    except P.NotDrawn as why:
        job.progress(0, 0, state="failed", message=str(why))
        raise FilmError(str(why)) from why
    stopped = job.should_stop()
    if not pictures:
        job.progress(0, 0, state="stopped" if stopped else "failed", message="no pictures")
        raise FilmError("The film was stopped before its first picture." if stopped else "No pictures to film.")
    fps = job.fps
    job.log(f"{look}: {len(pictures)} pictures; encoding")
    files: dict[str, str] = {}

    def images(target: tuple[int, int] | None = None) -> Iterable[Any]:
        for path in pictures:
            with Image.open(path) as im:
                im.load()
                img = im.convert("RGB")
            if target is not None and img.size != target:
                img = img.resize(target, Image.LANCZOS)
            yield img

    apng = folder / "film.png"
    write_apng(images(APNG_SIZE if size != APNG_SIZE else None), len(pictures), apng, fps)
    files["apng"] = os.path.relpath(apng, job.out)
    want_mp4 = "mp4" in job.formats or (not job.formats and mp4_available())
    want_gif = "gif" in job.formats or (not job.formats and not mp4_available())
    if want_mp4:
        write_mp4(images(), size, folder / "film.mp4", fps)
        files["mp4"] = os.path.relpath(folder / "film.mp4", job.out)
    if want_gif:
        palette = None
        if look == "storm3d":
            from . import storm3d

            palette = storm3d.gif_palette(pictures)
        write_gif(images(), folder / "film.gif", fps, palette=palette, keep=gif_keep(look))
        files["gif"] = os.path.relpath(folder / "film.gif", job.out)
    seconds = round(_time.perf_counter() - job.started, 2)
    record = {"look": look, "title": TITLES.get(look) or P.TITLES.get(look, look), "pictures": len(pictures),
              "fps": fps, "between": job.between if look == "swirl" else 0, "files": files,
              "frames": [os.path.relpath(p, job.out) for p in pictures], "size": list(size),
              "made_utc": _utc(), "seconds": seconds, "stopped": stopped, "note": blended_words,
              "mp4_available": mp4_available(), **({"facts": facts} if isinstance(facts, dict) else {})}
    index_path = job.out / FILMS / INDEX
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    if index.get("schema") != INDEX_SCHEMA:
        index = {"schema": INDEX_SCHEMA, "films": {}}
    index["films"][look] = record
    _write_json(index_path, index)
    job.progress(len(pictures), len(pictures), state="stopped" if stopped else "done",
                 message=f"{len(pictures)} pictures in {seconds:g} s")
    return record


def _derived(job: FilmJob) -> tuple[list[Path], tuple[int, int], dict[str, Any]]:
    from .facts import read_settings, run_facts
    from .frame import QuickFrame, frames_of
    from .ncread import discover

    layout = discover(job.target)
    lineage = frames_of(layout)
    if not lineage:
        raise FilmError(f"{job.target} has no published frames yet.")
    if layout.kind == "netcdf":
        settings = layout.settings
    else:
        settings, note = read_settings(layout)
        if note is not None:
            job.log(note)       # one plain line: without the settings a hurricane is drawn as a thunderstorm run
    frames_dir = job.folder / "frames"      # the pictures it had were set aside by make()
    if job.look == "storm3d":
        from . import storm3d

        return storm3d.film_pictures(job, frames_dir)
    first = QuickFrame.open(lineage[-1][2], key=lineage[-1][1], segment=lineage[-1][3])
    try:
        facts = run_facts(layout, first.config, first.dx, first.dy, label=job.label, settings=settings)
        extent, nxy, dx, height = first.extent, (first.nx, first.ny), first.dx, float(first.zh[0])
    finally:
        first.close()
    if job.look == "parcels":
        from . import parcels as PA

        if layout.kind == "netcdf":
            from .ncread import PARCELS_NOT_READ

            raise FilmError(PARCELS_NOT_READ)

        ended, _end = layout.ended()
        try:
            paths = PA.film_frames(layout, facts, extent, nxy, frames_dir, size=SWIRL_SIZE,
                                   should_stop=job.should_stop,
                                   progress=lambda done, total: job.progress(done, total))
        except PA.NoParcels as why:
            if not ended:
                raise FilmError("Parcels arrive when the run ends.") from why
            raise FilmError(f"{why} To follow them, set CM1's iprcl = 1 for the next run: in Advanced, Namelist, "
                            "or with gpuwm-cm1 edit RUNDIR --set iprcl=1 before it starts.") from why
        return paths, SWIRL_SIZE, {"parcel_times": len(paths)}
    # swirl
    W, H = SWIRL_SIZE
    box = P.fit_box(extent[0], extent[1], W // 2, int(H * 0.165), int(W * 0.62), int(H * 0.66))
    rng = P.swirl_range(dx, height, hurricane=_hurricane(facts))
    state = Swirl(box, P.noise(box.h, box.w, 5), P.noise(box.h, box.w, 5), rng)
    times = [row[0] for row in lineage]
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    interval = gaps[len(gaps) // 2] if gaps else None
    total = (len(lineage) - 1) * (job.between + 1) + 1
    paths: list[Path] = []
    previous = None
    number = 0
    for n, (t, key, path, segment) in enumerate(lineage):
        qf = QuickFrame.open(path, key=key, segment=segment)
        try:
            now = _sampled(qf, box)
        finally:
            qf.close()
        if previous is None:
            steps = [(0.0, now, False, 0.0)]
        else:
            span = now["t"] - previous["t"]
            steps = [((k / (job.between + 1)), None, True, span / (job.between + 1)) for k in range(1, job.between + 1)]
            steps.append((1.0, now, False, span / (job.between + 1)))
        for f, exact, blended, dt in steps:
            if job.should_stop():
                return paths, SWIRL_SIZE, {"saves": n, "pictures": len(paths), "interval_s": interval}
            fields = exact if exact is not None else _blend(previous, now, f)
            out = frames_dir / f"{number:05d}.png"
            swirl_picture(state, fields, dt, facts, extent, dx, blended, interval, out, height)
            paths.append(out)
            number += 1
            job.progress(number, total)
        previous = now
    return paths, SWIRL_SIZE, {"saves": len(lineage), "pictures": len(paths), "interval_s": interval,
                               "between": job.between}


__all__ = ["DERIVED", "FORMATS", "FilmError", "FilmJob", "FilmKept", "film_looks", "make", "mp4_available",
           "read_apng_frames", "remake_refused", "write_apng", "write_gif", "write_mp4"]
