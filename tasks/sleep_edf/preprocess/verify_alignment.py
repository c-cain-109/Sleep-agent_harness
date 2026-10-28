#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent alignment checker for a prepared Sleep-EDF .npz file.

Answers one question with evidence: **is every EEG epoch really paired with the
right sleep-stage label?** It deliberately does not import the preprocessing
module. Instead it re-implements, from scratch, the two things that could go
wrong:

1. *sample identity* - it computes the byte offset of epoch ``k`` in the EDF by
   hand (header size + k * data-record size + channel offset), decodes the int16
   samples, applies the EDF digital->physical formula, and asserts the result is
   float32-identical to ``x[i]``.
2. *label identity* - it regex-extracts the TAL annotations from the hypnogram
   and asserts that the stage covering ``[k*30s, (k+1)*30s)`` maps to ``y[i]``.

If both hold for every epoch, the pairing is proven rather than assumed.

Usage
-----
    python verify_alignment.py --psg SC4001E0-PSG.edf \\
        --hypnogram SC4001EC-Hypnogram.edf --npz out/SC4001E0.npz

Exit code 0 = every check passed, 1 = at least one mismatch.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Raw hypnogram text -> five-class id. Stages absent from this table must never
# appear in an output label, which is exactly what we assert.
STAGE_TO_LABEL: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
EXCLUDED_STAGES = {"Movement time", "Sleep stage ?"}

TAL_RE = re.compile(rb"\+([0-9.]+)\x15([0-9.]+)\x14([^\x14\x00]+)")


def _header_fields(body: bytes, n_signals: int, width: int, cursor: int):
    vals = [body[cursor + i * width: cursor + (i + 1) * width].decode("latin-1").strip()
            for i in range(n_signals)]
    return vals, cursor + width * n_signals


class RawEdf:
    """Minimal, self-contained EDF reader used only for verification."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            head = f.read(256)
            if len(head) < 256:
                raise SystemExit(f"{self.path.name}: shorter than an EDF header")
            self.n_signals = int(head[252:256].decode("ascii").strip())
            self.header_bytes = int(head[184:192].decode("ascii").strip())
            self.n_records = int(head[236:244].decode("ascii").strip())
            self.record_duration = float(head[244:252].decode("ascii").strip())
            body = f.read(self.header_bytes - 256)

        ns, cur = self.n_signals, 0
        self.labels, cur = _header_fields(body, ns, 16, cur)
        _, cur = _header_fields(body, ns, 80, cur)          # transducer
        _, cur = _header_fields(body, ns, 8, cur)           # physical dimension
        pmin_s, cur = _header_fields(body, ns, 8, cur)
        pmax_s, cur = _header_fields(body, ns, 8, cur)
        dmin_s, cur = _header_fields(body, ns, 8, cur)
        dmax_s, cur = _header_fields(body, ns, 8, cur)
        _, cur = _header_fields(body, ns, 80, cur)          # prefilter
        nsamp_s, cur = _header_fields(body, ns, 8, cur)

        self.phys_min = [float(v) for v in pmin_s]
        self.phys_max = [float(v) for v in pmax_s]
        self.dig_min = [int(v) for v in dmin_s]
        self.dig_max = [int(v) for v in dmax_s]
        self.n_samples = [int(v) for v in nsamp_s]
        self.record_bytes = sum(self.n_samples) * 2

    def channel(self, name: str) -> int:
        try:
            return self.labels.index(name)
        except ValueError:
            raise SystemExit(f"channel {name!r} not in {self.labels}")

    def read_epoch(self, k: int, channel: int) -> np.ndarray:
        """Physical values (uV) of epoch ``k`` for ``channel``, straight from bytes."""
        gain = ((self.phys_max[channel] - self.phys_min[channel])
                / (self.dig_max[channel] - self.dig_min[channel]))
        offset = self.phys_min[channel] - self.dig_min[channel] * gain
        ch_byte_offset = sum(self.n_samples[:channel]) * 2
        n = self.n_samples[channel]
        with open(self.path, "rb") as f:
            f.seek(self.header_bytes + k * self.record_bytes + ch_byte_offset)
            buf = f.read(n * 2)
        digital = np.frombuffer(buf, dtype="<i2").astype(np.float64)
        return digital * gain + offset


def read_annotations(path: Path) -> List[Tuple[float, float, str]]:
    with open(path, "rb") as f:
        head = f.read(256)
        header_bytes = int(head[184:192].decode("ascii").strip())
        f.seek(header_bytes)
        blob = f.read()
    out = []
    for m in TAL_RE.finditer(blob):
        out.append((float(m.group(1)), float(m.group(2)),
                    m.group(3).decode("latin-1").strip()))
    return out


def label_at(sec: float, anns: List[Tuple[float, float, str]]) -> Tuple[Optional[int], str]:
    for onset, dur, text in anns:
        if onset - 1e-6 <= sec < onset + dur - 1e-6:
            return STAGE_TO_LABEL.get(text), text
    return None, "<no annotation>"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psg", required=True, help="source *-PSG.edf")
    ap.add_argument("--hypnogram", required=True, help="source *-Hypnogram.edf")
    ap.add_argument("--npz", required=True, help="the prepared .npz to check")
    ap.add_argument("--channel", default="EEG Fpz-Cz")
    ap.add_argument("--max-report", type=int, default=10,
                    help="how many mismatches to print before truncating")
    args = ap.parse_args(argv)

    psg = RawEdf(Path(args.psg))
    ch = psg.channel(args.channel)
    anns = read_annotations(Path(args.hypnogram))
    if not anns:
        raise SystemExit("no annotations parsed from the hypnogram")

    with np.load(args.npz) as z:
        x, y = z["x"], z["y"]
        eidx = z["epoch_index"] if "epoch_index" in z.files else None
        epoch_duration = int(z["epoch_duration"]) if "epoch_duration" in z.files else 30

    if eidx is None:
        raise SystemExit("the npz has no 'epoch_index' key; cannot locate epochs in the EDF")

    print(f"EDF    : {psg.path.name}")
    print(f"  channel            : {psg.labels[ch]!r} (index {ch})")
    print(f"  samples per record : {psg.n_samples[ch]}")
    print(f"  record duration    : {psg.record_duration}s -> "
          f"{psg.n_samples[ch] / psg.record_duration:.3f} Hz")
    print(f"  data records       : {psg.n_records} "
          f"({psg.n_records * psg.record_duration:.0f}s of signal)")
    print(f"annotations          : {len(anns)} TAL entries")
    print(f"npz                  : x={x.shape} {x.dtype}  y={y.shape} {y.dtype}")
    print()

    bad_signal = []
    bad_label = []
    for i in range(len(x)):
        k = int(eidx[i])
        reference = psg.read_epoch(k, ch).astype(np.float32)
        if not np.array_equal(reference, x[i]):
            bad_signal.append((i, k))
        expected, stage = label_at(k * epoch_duration, anns)
        if expected != int(y[i]):
            bad_label.append((i, k, int(y[i]), expected, stage))

    n = len(x)
    print("=" * 66)
    print(f"sample identity : {n - len(bad_signal)}/{n} epochs match the raw EDF bytes")
    print(f"label identity  : {n - len(bad_label)}/{n} epochs match the hypnogram")
    print("=" * 66)
    for rec in bad_signal[: args.max_report]:
        print(f"  SIGNAL MISMATCH i={rec[0]} epoch_index={rec[1]}")
    for rec in bad_label[: args.max_report]:
        print(f"  LABEL  MISMATCH i={rec[0]} epoch={rec[1]} "
              f"npz={rec[2]} hypnogram={rec[3]} ({rec[4]})")

    # annotation boundaries must not fall inside an epoch
    off_grid = [(o, d, s) for o, d, s in anns
                if abs(o / epoch_duration - round(o / epoch_duration)) > 1e-6
                or abs(d / epoch_duration - round(d / epoch_duration)) > 1e-6]
    print(f"annotations on the {epoch_duration}s grid : "
          f"{len(anns) - len(off_grid)}/{len(anns)}")
    for item in off_grid[: args.max_report]:
        print(f"  off-grid annotation: onset={item[0]} dur={item[1]} {item[2]!r}")

    ok = not bad_signal and not bad_label and not off_grid
    print()
    print("RESULT:", "PASS - every epoch's samples and label are correctly paired" if ok
          else "FAIL - see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
