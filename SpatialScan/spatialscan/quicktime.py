"""A small QuickTime / ISO-BMFF atom reader for Apple *spatial video*.

Apple spatial video (iPhone 15 Pro, Vision Pro) is a QuickTime ``.mov`` whose
video track is **MV-HEVC** — a base layer (one eye) plus a dependent layer (the
other eye) in a single elementary stream. The stereo geometry we need to turn
disparity into *metric* depth lives in the container as extension atoms, not in
the pixels:

* ``vexu`` (Video Extended Usage) sits inside the ``hvc1``/``hev1`` sample
  entry and carries the stereo description.
* ``vexu > eyes > stri`` — Stereo view Information: which eyes are present and
  whether they are stored left/right reversed.
* ``vexu > eyes > cams > blin`` — **baseline** (camera separation) in
  *micrometres*, stored big-endian.
* ``vexu > proj`` / ``cmfx > hfov`` — horizontal field of view in
  *milli-degrees* (thousandths of a degree).

This module walks the box tree (sizes and nesting are fully specified by
ISO-BMFF) and pulls those scalars out. Byte layouts of the leaf spatial atoms
are read *best-effort* with sanity clamps — if a field is missing or out of
range the loader falls back to user-supplied / iPhone-default values, so a
wrong guess never silently corrupts the reconstruction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Boxes whose payload is itself a sequence of child boxes.
_CONTAINERS = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf",
    b"vexu", b"eyes", b"cams", b"cmfx", b"proj", b"hero",
}
# FullBox containers: 4 header bytes (version+flags) precede the child boxes.
_FULL_CONTAINERS = {b"meta": 4, b"stsd": 8}
# Sample-entry boxes hold a fixed VisualSampleEntry header, then child boxes.
_SAMPLE_ENTRIES = {b"hvc1", b"hev1", b"avc1", b"mv-h", b"hvce"}
_VISUAL_SAMPLE_ENTRY_HEADER = 78  # SampleEntry(8) + VisualSampleEntry(70)


@dataclass
class Atom:
    """One parsed box: its type, byte range, and either children or raw data."""

    type: bytes
    offset: int          # start of the box header in the file
    size: int            # total box size including header
    header_size: int
    data: bytes = b""    # raw payload for leaf atoms
    children: list["Atom"] = field(default_factory=list)

    @property
    def fourcc(self) -> str:
        return self.type.decode("latin-1", "replace")

    def find(self, fourcc: str) -> "Atom | None":
        """First descendant (any depth) with this type, or None."""
        for a in self.walk():
            if a is not self and a.fourcc == fourcc:
                return a
        return None

    def find_all(self, fourcc: str) -> list["Atom"]:
        return [a for a in self.walk() if a is not self and a.fourcc == fourcc]

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def _parse_boxes(buf: bytes, start: int, end: int, depth: int = 0) -> list[Atom]:
    atoms: list[Atom] = []
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", buf, pos)[0]
        btype = buf[pos + 4:pos + 8]
        header = 8
        if size == 1:  # 64-bit largesize
            if pos + 16 > end:
                break
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            header = 16
        elif size == 0:  # extends to the end of the enclosing box
            size = end - pos
        if size < header or pos + size > end:
            break  # truncated / not a real box — stop scanning this level

        payload = pos + header
        payload_end = pos + size
        atom = Atom(type=btype, offset=pos, size=size, header_size=header)

        if btype in _CONTAINERS and depth < 12:
            atom.children = _parse_boxes(buf, payload, payload_end, depth + 1)
        elif btype in _FULL_CONTAINERS and depth < 12:
            skip = _FULL_CONTAINERS[btype]
            atom.children = _parse_boxes(buf, payload + skip, payload_end, depth + 1)
        elif btype in _SAMPLE_ENTRIES and depth < 12:
            inner = payload + _VISUAL_SAMPLE_ENTRY_HEADER
            if inner < payload_end:
                atom.children = _parse_boxes(buf, inner, payload_end, depth + 1)
            atom.data = buf[payload:payload_end]
        else:
            atom.data = buf[payload:payload_end]

        atoms.append(atom)
        pos = payload_end
    return atoms


def parse_atoms(buf: bytes) -> Atom:
    """Parse the whole file into a synthetic ``root`` atom holding top boxes."""
    root = Atom(type=b"root", offset=0, size=len(buf), header_size=0)
    root.children = _parse_boxes(buf, 0, len(buf))
    return root


# ---------------------------------------------------------------------------
# Spatial-metadata extraction
# ---------------------------------------------------------------------------

@dataclass
class SpatialMetadata:
    """Stereo geometry recovered from the container (best-effort)."""

    is_mv_hevc: bool = False
    baseline_m: float | None = None      # camera separation, metres
    hfov_deg: float | None = None        # horizontal field of view, degrees
    has_left_eye: bool = True
    has_right_eye: bool = True
    eyes_reversed: bool = False          # stored right-then-left instead of L/R
    source_boxes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"MV-HEVC={self.is_mv_hevc}"]
        if self.baseline_m is not None:
            parts.append(f"baseline={self.baseline_m * 1000:.2f}mm")
        if self.hfov_deg is not None:
            parts.append(f"hfov={self.hfov_deg:.2f}deg")
        if self.eyes_reversed:
            parts.append("eyes_reversed")
        parts.append("boxes=" + ",".join(self.source_boxes) if self.source_boxes else "boxes=none")
        return " ".join(parts)


def _read_u32(data: bytes, off: int = 0) -> int | None:
    if len(data) >= off + 4:
        return struct.unpack_from(">I", data, off)[0]
    return None


def extract_spatial_metadata(buf: bytes) -> SpatialMetadata:
    """Pull baseline / hFOV / eye layout out of a spatial-video ``.mov`` blob."""
    root = parse_atoms(buf)
    meta = SpatialMetadata()

    vexu = root.find("vexu")
    # MV-HEVC is signalled by a layered-HEVC config (lhvC / hvcE) or a vexu box.
    if root.find("lhvC") or root.find("hvcE") or vexu is not None:
        meta.is_mv_hevc = True

    # blin lives in vexu>eyes>cams; hfov is a sibling of vexu in the sample
    # entry. Search the vexu subtree first, then the whole file — these
    # fourccs are rare, and the sanity clamps below reject false positives.
    scopes = [s for s in (vexu, root) if s is not None]

    blin = _first(scopes, "blin")
    if blin is not None:
        # baseline, big-endian; a leading version/flags word is optional, and
        # some writers store it 64-bit, so probe a few offsets.
        for off in (0, 4):
            micro = _read_u32(blin.data, off)
            if micro and 500 <= micro <= 500_000:  # 0.5mm .. 500mm sane range
                meta.baseline_m = micro / 1_000_000.0
                meta.source_boxes.append("blin")
                break

    hfov = _first(scopes, "hfov") or _first(scopes, "dfov")
    if hfov is not None:
        for off in (0, 4):
            milli = _read_u32(hfov.data, off)
            if milli and 20_000 <= milli <= 160_000:  # 20 .. 160 degrees
                meta.hfov_deg = milli / 1000.0
                meta.source_boxes.append(hfov.fourcc)
                break

    stri = _first(scopes, "stri")
    if stri is not None and stri.data:
        # StereoViewInformationBox (ISO/IEC 14496-12): a FullBox (4-byte
        # version/flags) then one flags byte, low bits: has_left(0x01),
        # has_right(0x02), has_additional(0x04), eye_views_reversed(0x08).
        flags = stri.data[4] if len(stri.data) > 4 else stri.data[-1]
        if flags & 0x03:  # at least one eye-presence bit set
            meta.has_left_eye = bool(flags & 0x01)
            meta.has_right_eye = bool(flags & 0x02)
        meta.eyes_reversed = bool(flags & 0x08)
        meta.source_boxes.append("stri")

    return meta


def _first(scopes, fourcc: str) -> "Atom | None":
    """First match for ``fourcc`` across the given scopes, in order."""
    for s in scopes:
        hit = s.find(fourcc)
        if hit is not None:
            return hit
    return None
