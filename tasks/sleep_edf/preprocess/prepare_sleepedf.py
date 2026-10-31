#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sleep-EDF (Expanded / Sleep Cassette) preprocessing for Sleep-Harness.

Builds 30 s, 5-class (W / N1 / N2 / N3 / REM) single-channel EEG epochs from a
paired ``*-PSG.edf`` + ``*-Hypnogram.edf`` recording, and writes the artifacts
consumed by the PyTorch TinySleepNet trainer/evaluator.

Design goals
------------
* No hard-coded absolute paths: input/output locations come from the CLI or a
  config file.
* Non-destructive: source EDF files are opened read-only and never modified;
  an existing output directory is never wiped.
* Auditable: every excluded epoch and every anomaly is counted and reported in
  ``dataset_manifest.json`` and ``preprocessing.log``.
* Reproducible: no randomness, deterministic ``.npz`` bytes, fixed float32
  storage.
* Batch ready: scans a directory for all PSG/Hypnogram pairs, and can emit a
  subject-level train/val/test split for multi-subject experiments.

Epoch / label alignment rule (see README for the long version)
--------------------------------------------------------------
The epoch grid is anchored at the first EEG sample of the PSG record. Epoch ``k``
covers ``[k * 30 s, (k + 1) * 30 s)``. A hypnogram annotation with onset ``o``
and duration ``d`` labels exactly the epochs ``[o/30, (o+d)/30)`` and is only
accepted when both ``o`` and ``d`` are exact multiples of the epoch duration, so
that an annotation boundary can never fall inside an epoch. Epochs with no
covering annotation are excluded. This is the same "annotation fully covers the
epoch" contract used by the reference TinySleepNet implementation.

Usage
-----
    python prepare_sleepedf.py --input-dir  <edf_dir> --output-dir <out_dir>
    python prepare_sleepedf.py --psg F.edf --hypnogram H.edf --output-dir <out_dir>
    python prepare_sleepedf.py --config config.yaml

Author: Sleep-Harness data pipeline
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import platform
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.0.0"

LOG = logging.getLogger("prepare_sleepedf")
# ---------------------------------------------------------------------------
# Constants: stage tables
# ---------------------------------------------------------------------------

#: Public five-class label ids (what ends up in ``y``).
LABEL_NAMES: Dict[int, str] = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

#: Raw hypnogram text -> internal code. Codes 0-4 are the five classes, 5 and 6
#: are placeholders used only for bookkeeping/edge trimming and are dropped
#: before the dataset is written. S3 and S4 are both mapped to N3 (AASM rule).
DEFAULT_LABEL_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

#: Stages that are recognised but deliberately kept out of the five-class task.
DEFAULT_EXCLUDED_STAGES: Dict[str, int] = {
    "Movement time": 5,
    "Sleep stage ?": 6,
}

CODE_MOVEMENT = 5
CODE_UNKNOWN = 6
CODE_UNCOVERED = -1  # epoch not covered by any annotation

DEFAULTS: Dict[str, Any] = {
    # I/O
    "input_dir": None,
    "output_dir": None,
    "psg": None,
    "hypnogram": None,
    # signal / epoch definition
    "channel": "EEG Fpz-Cz",
    "target_fs": 100,
    "epoch_duration": 30,
    # labels
    "label_map": dict(DEFAULT_LABEL_MAP),
    "excluded_stages": dict(DEFAULT_EXCLUDED_STAGES),
    # alignment / cleaning policy
    "trim_edge_wake": True,
    "edge_wake_epochs": 60,
    "on_annotation_gap": "drop",      # drop | error
    "on_stage_conflict": "error",     # error | drop
    "drop_nonfinite_epochs": True,
    # reader / runtime
    "reader": "auto",                 # auto | mne | native
    "overwrite": False,
    "strict": False,
    "log_level": "INFO",
    # subject split
    "make_split": False,
    "split_seed": 42,
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "subject_id_regex": None,
    # naming
    "psg_suffix": "-PSG.edf",
    "hypnogram_suffix": "-Hypnogram.edf",
}

NPZ_REQUIRED_KEYS = (
    "x", "y", "fs", "ch_label", "epoch_duration", "record_id", "subject_id",
)

# ---------------------------------------------------------------------------
# Zero-argument defaults
# ---------------------------------------------------------------------------
# Pressing "Run" in an IDE passes no arguments. Rather than failing with a bare
# "output dir required", we resolve conventional locations relative to the
# repository and the current working directory, and only complain (with a full
# explanation of what was searched) when the recordings really cannot be found.
#
# Nothing here is an absolute path: every candidate is built from the script's
# own location, the CWD, or an environment variable.

#: Environment variable that can point straight at the EDF folder.
ENV_DATA_DIR = "SLEEPEDF_DATA_DIR"

#: Output folder created under the repository root when --output-dir is omitted.
DEFAULT_OUTPUT_RELPATH = Path("outputs") / "prepared"

#: Input folders probed, in order, under both the CWD and the repository root.
DEFAULT_INPUT_RELPATHS = (
    Path("data") / "sleep-edf",
    Path("data") / "sleep_edf",
    Path("data") / "sleepedf",
    Path("data"),
    Path("."),                    # e.g. EDFs dropped straight into the project
)


def project_root() -> Path:
    """Repository root inferred from this file's location.

    ``<root>/tasks/sleep_edf/prepare_sleepedf.py`` -> ``<root>``. Falls back to
    the file's own directory if the layout is different, so a single-file copy
    still gets a stable, non-absolute base.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tasks" / "sleep_edf").is_dir():
            return parent
    return here.parent


def _contains_psg(dirpath: Path, max_depth: int = 2) -> bool:
    """True if ``dirpath`` holds at least one ``*-PSG.edf`` (shallow search)."""
    try:
        if any(dirpath.glob("*PSG.edf")):
            return True
        if any(dirpath.glob("*psg.edf")):
            return True
    except OSError:
        return False
    if max_depth <= 0:
        return False
    try:
        for child in dirpath.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if _contains_psg(child, max_depth - 1):
                    return True
    except OSError:
        return False
    return False


def default_candidate_dirs(bases: Optional[Sequence[Path]] = None) -> List[Path]:
    """All directories that will be probed for recordings, in priority order.

    ``bases`` defaults to the current working directory plus the repository root;
    tests pass their own bases so nothing depends on where pytest/unittest runs.
    """
    candidates: List[Path] = []

    env_value = os.environ.get(ENV_DATA_DIR)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    if bases is None:
        bases = [Path.cwd()]
        root = project_root()
        if root != Path.cwd():
            bases = [Path.cwd(), root]
    bases = list(bases)

    for base in bases:
        for rel in DEFAULT_INPUT_RELPATHS:
            candidates.append(base / rel)

    # de-duplicate while preserving order
    seen: set = set()
    unique: List[Path] = []
    for c in candidates:
        try:
            key = c.resolve()
        except OSError:
            key = c
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def find_default_input_dir(bases: Optional[Sequence[Path]] = None
                           ) -> Tuple[Optional[Path], List[Path]]:
    """Locate the EDF folder without any arguments.

    Returns ``(found_dir_or_None, searched_candidates)``.
    """
    candidates = default_candidate_dirs(bases)
    for cand in candidates:
        try:
            if cand.is_dir() and _contains_psg(cand):
                return cand, candidates
        except OSError:
            continue
    return None, candidates


def _usage_hint(searched: List[Path]) -> str:
    """Build the action-oriented help block shown when no data folder is found."""
    lines = [
        "",
        "This script needs to know where your EDF files are.",
        "",
        "Three ways to tell it, pick one:",
        "",
        "  1) Put the files in a conventional folder and just press Run again:",
        "       <project>/data/sleep-edf/<your>*-PSG.edf",
        "       <project>/data/sleep-edf/<your>*-Hypnogram.edf",
        "",
        "  2) Pass the folder explicitly (PyCharm: Run > Edit Configurations",
        "     > Script parameters):",
        '       --input-dir "D:\\path\\to\\edf_folder"',
        "",
        "  3) Set an environment variable (PyCharm: Run > Edit Configurations",
        "     > Environment variables):",
        f'       {ENV_DATA_DIR}=D:\\path\\to\\edf_folder',
        "",
        "  Or process a single pair directly:",
        '       --psg "X-PSG.edf" --hypnogram "X-Hypnogram.edf"',
        "",
        "Output defaults to <project>/outputs/prepared when --output-dir is omitted,",
        "so you normally do NOT need to pass --output-dir at all.",
        "",
        "Folders that were searched for *-PSG.edf:",
    ]
    for cand in searched:
        try:
            if not cand.is_dir():
                state = "missing"
            elif _contains_psg(cand):
                state = "has EDF"
            else:
                state = "no EDF "
        except OSError:
            state = "missing"
        lines.append(f"  [{state}] {cand}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base class for fatal, user-facing pipeline errors."""


class ConfigError(PipelineError):
    """Bad CLI/config combination or unsafe output directory."""


class UsageError(ConfigError):
    """The invocation itself is incomplete: we could not work out where the data is.

    Carries a human-readable ``hint`` block so the caller can print *what to do
    next* instead of a bare one-line error. This is the case that hits people who
    press "Run" in an IDE without adding arguments.
    """

    def __init__(self, message: str, hint: str = "", searched: Optional[List[str]] = None):
        super().__init__(message)
        self.hint = hint
        self.searched = searched or []


class EdfFormatError(PipelineError):
    """Malformed or unsupported EDF/EDF+ file."""


class RecordError(PipelineError):
    """A single record failed validation; the batch continues."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks (works for big EDFs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_edf_float(text: str, default: float = 0.0) -> float:
    text = text.strip()
    return float(text) if text else default


def _parse_edf_int(text: str, default: int = 0) -> int:
    text = text.strip()
    return int(float(text)) if text else default


def _edf_datetime(date_str: str, time_str: str, fallback_year: int = 1985) -> datetime:
    """Parse the ``dd.mm.yy`` / ``hh.mm.ss`` pair stored in the EDF fixed header.

    EDF (1985) uses a two-digit year. Following the EDF+ convention the window
    is 1985-2084, i.e. yy >= 85 means 19yy, otherwise 20yy.
    """
    date_str = date_str.strip().replace("-", ".")
    time_str = time_str.strip().replace("-", ":")
    parts = re.split(r"[.:]", date_str)
    tparts = re.split(r"[.:]", time_str)
    if len(parts) != 3:
        raise EdfFormatError(f"Unparsable EDF start date {date_str!r}")
    day, month, year2 = (int(p) for p in parts)
    year = 1900 + year2 if year2 >= 85 else 2000 + year2
    hour = int(tparts[0]) if len(tparts) > 0 and tparts[0] else 0
    minute = int(tparts[1]) if len(tparts) > 1 and tparts[1] else 0
    second = int(float(tparts[2])) if len(tparts) > 2 and tparts[2] else 0
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:  # pragma: no cover - corrupt header
        raise EdfFormatError(
            f"Invalid EDF start datetime {date_str!r} {time_str!r}: {exc}"
        ) from exc


def _fmt_seconds(value: float) -> str:
    return f"{value:.3f}s"


# ---------------------------------------------------------------------------
# EDF reading (header parser is dependency-free on purpose)
# ---------------------------------------------------------------------------

@dataclass
class EdfHeader:
    """Everything we need from an EDF fixed + variable header."""

    path: Path
    version: str
    patient: str
    recording: str
    start_datetime: datetime
    header_bytes: int
    n_records: int
    record_duration: float
    n_signals: int
    labels: List[str]
    transducer: List[str]
    phys_dim: List[str]
    phys_min: List[float]
    phys_max: List[float]
    dig_min: List[int]
    dig_max: List[int]
    prefilter: List[str]
    n_samples: List[int]
    reserved: str
    file_size: int

    # -- derived ----------------------------------------------------------
    @property
    def is_annotation_only(self) -> bool:
        return any("annotation" in lab.lower() for lab in self.labels)

    @property
    def record_bytes(self) -> int:
        return int(sum(self.n_samples) * 2)

    @property
    def sfreq(self) -> List[float]:
        if self.record_duration <= 0:
            return [0.0 for _ in self.n_samples]
        return [n / self.record_duration for n in self.n_samples]

    def signal_duration(self) -> float:
        """Total seconds of signal actually present in the file."""
        if self.record_duration <= 0:
            return 0.0
        return self.n_records * self.record_duration

    def channel_index(self, name: str) -> int:
        for idx, lab in enumerate(self.labels):
            if lab == name:
                return idx
        # tolerate trailing whitespace / case differences but never guess blindly
        norm = name.strip().lower()
        for idx, lab in enumerate(self.labels):
            if lab.strip().lower() == norm:
                return idx
        raise RecordError(
            f"Channel {name!r} not found in {self.path.name}. "
            f"Available channels: {self.labels}"
        )


def read_edf_header(path: Path) -> EdfHeader:
    """Parse the EDF/EDF+ header. Works for annotation-only files too."""
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as handle:
        fixed = handle.read(256)
        if len(fixed) < 256:
            raise EdfFormatError(f"{path.name}: file shorter than the 256-byte EDF header")
        try:
            n_signals = int(fixed[252:256].decode("ascii").strip())
            header_bytes = int(fixed[184:192].decode("ascii").strip())
            n_records = int(fixed[236:244].decode("ascii").strip())
            record_duration = float(fixed[244:252].decode("ascii").strip())
        except ValueError as exc:
            raise EdfFormatError(f"{path.name}: unreadable EDF header fields ({exc})") from exc

        if n_signals <= 0:
            raise EdfFormatError(f"{path.name}: header declares {n_signals} signals")

        body = handle.read(max(0, header_bytes - 256))

    cursor = 0

    def take(width: int) -> List[str]:
        nonlocal cursor
        values = [
            body[cursor + i * width: cursor + (i + 1) * width].decode("latin-1").strip()
            for i in range(n_signals)
        ]
        cursor += width * n_signals
        return values

    labels = take(16)
    transducer = take(80)
    phys_dim = take(8)
    phys_min = [_parse_edf_float(v) for v in take(8)]
    phys_max = [_parse_edf_float(v) for v in take(8)]
    dig_min = [_parse_edf_int(v) for v in take(8)]
    dig_max = [_parse_edf_int(v) for v in take(8)]
    prefilter = take(80)
    n_samples = [_parse_edf_int(v) for v in take(8)]
    take(32)  # per-signal reserved

    if n_records < 0:  # -1 means "unknown": derive it from the file size
        record_bytes = sum(n_samples) * 2
        if record_bytes > 0:
            n_records = (file_size - header_bytes) // record_bytes

    header = EdfHeader(
        path=path,
        version=fixed[0:8].decode("latin-1").strip("\x00 "),
        patient=fixed[8:88].decode("latin-1").strip(),
        recording=fixed[88:168].decode("latin-1").strip(),
        start_datetime=_edf_datetime(
            fixed[168:176].decode("latin-1"),
            fixed[176:184].decode("latin-1"),
        ),
        header_bytes=header_bytes,
        n_records=int(n_records),
        record_duration=record_duration,
        n_signals=n_signals,
        labels=labels,
        transducer=transducer,
        phys_dim=phys_dim,
        phys_min=phys_min,
        phys_max=phys_max,
        dig_min=dig_min,
        dig_max=dig_max,
        prefilter=prefilter,
        n_samples=n_samples,
        reserved=fixed[192:236].decode("latin-1").strip("\x00 "),
        file_size=file_size,
    )

    expected = 256 * (n_signals + 1)
    if header.header_bytes != expected:
        LOG.warning(
            "%s: header declares %d bytes but %d signals imply %d bytes",
            path.name, header.header_bytes, n_signals, expected,
        )
    return header


# -- annotations ------------------------------------------------------------

TAL_SEP_FIELD = b"\x14"
TAL_SEP_TEXT = b"\x15"
TAL_END = b"\x14\x00"


@dataclass(frozen=True)
class Annotation:
    onset: float
    duration: float
    text: str
    record: int = 0


def _tal_number(raw: bytes) -> Optional[float]:
    text = raw.replace(b"\x00", b"").decode("latin-1").strip().lstrip("+")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_tal_block(block: bytes, record_idx: int = 0) -> List[Annotation]:
    """Decode one EDF+ 'EDF Annotations' data record (a list of TALs).

    The canonical EDF+ layout is ``+onset \\x15 [+duration \\x14] text \\x14 \\x00``
    (``\\x15`` separates the onset from the payload, ``\\x14`` separates the
    following fields). A minority of writers emit ``+onset \\x14 duration \\x15
    text`` instead, so both are accepted. Entries with no text payload are
    timekeeping markers and are ignored.
    """
    found: List[Annotation] = []
    for tal in block.split(TAL_END):
        tal = tal.replace(b"\x00", b"")
        if not tal:
            continue

        if TAL_SEP_TEXT in tal:
            head, _, rest = tal.partition(TAL_SEP_TEXT)
            fields = rest.split(TAL_SEP_FIELD)
            onset = _tal_number(head)
            if onset is None:
                LOG.warning("Skipping TAL with bad onset (record %d): %r", record_idx, tal[:40])
                continue
            if len(fields) >= 2:
                duration = _tal_number(fields[0]) or 0.0
                text = TAL_SEP_FIELD.join(fields[1:]).decode("latin-1").strip()
            else:
                duration, text = 0.0, fields[0].decode("latin-1").strip()
        else:
            fields = tal.split(TAL_SEP_FIELD)
            if len(fields) < 3:  # timekeeping marker only
                continue
            onset = _tal_number(fields[0])
            if onset is None:
                continue
            duration = _tal_number(fields[1]) or 0.0
            text = TAL_SEP_FIELD.join(fields[2:]).decode("latin-1").strip()

        if not text:
            continue
        found.append(Annotation(onset=onset, duration=duration, text=text, record=record_idx))
    return found


def fix_missing_durations(annotations: List[Annotation],
                          total_duration: Optional[float] = None) -> List[Annotation]:
    """Fill in zero/missing annotation durations from the next onset.

    Some hypnograms omit the duration field entirely. Rather than dropping those
    epochs we infer the span from the following annotation (and from the signal
    length for the final entry), and log every inference so it is auditable.
    """
    if not annotations:
        return annotations
    ordered = sorted(annotations, key=lambda a: a.onset)
    fixed: List[Annotation] = []
    for i, ann in enumerate(ordered):
        if ann.duration > 0:
            fixed.append(ann)
            continue
        nxt = ordered[i + 1].onset if i + 1 < len(ordered) else total_duration
        inferred = (nxt - ann.onset) if nxt is not None else 0.0
        if inferred <= 0:
            LOG.warning("Dropping zero-duration annotation %r at %ss (no next onset)",
                        ann.text, ann.onset)
            continue
        LOG.info("Inferred missing duration for %r at %ss: %.1fs", ann.text, ann.onset, inferred)
        fixed.append(Annotation(onset=ann.onset, duration=inferred, text=ann.text,
                                record=ann.record))
    return fixed


def read_edf_annotations(path: Path,
                         total_duration: Optional[float] = None
                         ) -> Tuple[EdfHeader, List[Annotation]]:
    """Read all annotations from an EDF+ file that carries an annotation channel.

    ``total_duration`` (the PSG length in seconds) is only used to infer the
    duration of a trailing annotation that has no explicit duration field.
    """
    path = Path(path)
    header = read_edf_header(path)
    if not header.is_annotation_only:
        raise EdfFormatError(
            f"{path.name} has no 'EDF Annotations' channel; it does not look like "
            f"a hypnogram file (channels: {header.labels})"
        )

    ann_idx = next(i for i, lab in enumerate(header.labels) if "annotation" in lab.lower())
    before = int(sum(header.n_samples[:ann_idx]) * 2)
    width = int(header.n_samples[ann_idx] * 2)

    annotations: List[Annotation] = []
    with open(path, "rb") as handle:
        for rec in range(header.n_records):
            handle.seek(header.header_bytes + rec * header.record_bytes + before)
            block = handle.read(width)
            if not block:
                break
            annotations.extend(_parse_tal_block(block, rec))

    if not annotations:
        raise EdfFormatError(f"{path.name}: no annotations found")

    annotations.sort(key=lambda a: a.onset)
    annotations = fix_missing_durations(annotations, total_duration)
    if not annotations:
        raise EdfFormatError(f"{path.name}: every annotation had a zero duration")
    return header, annotations


# -- signal ----------------------------------------------------------------

def read_signal_native(header: EdfHeader, channel: str) -> np.ndarray:
    """Read one channel as physical values (microvolts) using only NumPy.

    Samples are int16 little-endian, interleaved per data record, so the channel
    is fetched record by record. The digital -> physical conversion is the EDF
    formula ``phys_min + (dig - dig_min) * (phys_max - phys_min) / (dig_max - dig_min)``,
    which is exactly what pyedflib (hence the reference TinySleepNet code) does.
    """
    idx = header.channel_index(channel)
    n_samp = header.n_samples[idx]
    offset = int(sum(header.n_samples[:idx]) * 2)
    gain = (header.phys_max[idx] - header.phys_min[idx]) / (header.dig_max[idx] - header.dig_min[idx])
    intercept = header.phys_min[idx] - header.dig_min[idx] * gain

    blocks = []
    with open(header.path, "rb") as handle:
        for rec in range(header.n_records):
            handle.seek(header.header_bytes + rec * header.record_bytes + offset)
            raw = handle.read(n_samp * 2)
            if len(raw) < n_samp * 2:
                LOG.warning("%s: truncated data record %d", header.path.name, rec)
                break
            blocks.append(raw)

    if not blocks:
        raise RecordError(f"{header.path.name}: no samples could be read for {channel!r}")

    digital = np.frombuffer(b"".join(blocks), dtype="<i2").astype(np.float64)
    return digital * gain + intercept


def read_signal_mne(path: Path, channel: str) -> Tuple[np.ndarray, float]:
    """Read one channel via MNE, returned in microvolts (MNE stores volts)."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("MNE reader requested but MNE is not installed") from exc

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        if channel not in raw.ch_names:
            raise RecordError(
                f"Channel {channel!r} not found in {path.name}. "
                f"Available channels: {raw.ch_names}"
            )
        data = raw.get_data(picks=[channel])[0] * 1e6  # volts -> microvolts
        sfreq = float(raw.info["sfreq"])
    finally:
        raw.close()
    return data.astype(np.float64, copy=False), sfreq


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling, used only when the recording is not at ``fs_out``."""
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from scipy.signal import resample_poly  # scipy ships with MNE

    g = math.gcd(int(round(fs_out)), int(round(fs_in)))
    up, down = int(round(fs_out)) // g, int(round(fs_in)) // g
    LOG.info("Resampling signal: %.3f Hz -> %.3f Hz (up=%d, down=%d)",
             fs_in, fs_out, up, down)
    return resample_poly(x, up, down)


# ---------------------------------------------------------------------------
# Annotation -> per-epoch code array
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    codes: np.ndarray                     # per-epoch internal code, -1 = uncovered
    stages: np.ndarray                    # per-epoch raw annotation text ("" if none)
    ann_table: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    n_truncated: int = 0


def build_epoch_codes(
    annotations: Sequence[Annotation],
    n_epochs: int,
    epoch_duration: float,
    label_map: Dict[str, int],
    excluded_stages: Dict[str, int],
    on_gap: str = "drop",
    on_conflict: str = "error",
) -> AlignmentResult:
    """Project hypnogram annotations onto the fixed 30 s epoch grid."""
    codes = np.full(n_epochs, CODE_UNCOVERED, dtype=np.int16)
    stages = np.full(n_epochs, "", dtype=object)
    table: List[Dict[str, Any]] = []
    issues: List[str] = []
    n_truncated = 0
    covered_until = 0.0

    for ann in annotations:
        stage = ann.text.strip()
        if stage in label_map:
            code = int(label_map[stage])
        elif stage in excluded_stages:
            code = int(excluded_stages[stage])
        else:
            raise RecordError(
                f"Unmapped hypnogram stage {stage!r} (onset {ann.onset}s, duration "
                f"{ann.duration}s). Unknown or unscored stages must never be mapped "
                f"silently; extend label_map/excluded_stages in the config if needed."
            )

        if ann.duration <= 0:
            issues.append(f"Zero-duration annotation {stage!r} at {ann.onset}s ignored")
            continue

        # annotate exact grid alignment: boundaries must land on epoch edges
        start_f = ann.onset / epoch_duration
        n_f = ann.duration / epoch_duration
        start_e, n_e = int(round(start_f)), int(round(n_f))
        if abs(start_f - start_e) > 1e-6 or abs(n_f - n_e) > 1e-6:
            issues.append(
                f"Annotation {stage!r} at onset {ann.onset}s dur {ann.duration}s is not "
                f"aligned to the {epoch_duration:g}s epoch grid "
                f"(start={start_f:.4f}, n={n_f:.4f} epochs) -> epochs excluded"
            )
            table.append({
                "onset": ann.onset, "duration": ann.duration, "stage": stage,
                "code": code, "epoch_start": None, "n_epochs": 0, "status": "misaligned",
            })
            continue

        if ann.onset > covered_until + 1e-6:
            gap = ann.onset - covered_until
            msg = f"Annotation gap of {gap:.1f}s before {stage!r} at {ann.onset}s"
            issues.append(msg)
            if on_gap == "error":
                raise RecordError(f"Hypnogram is not contiguous: {msg}")
            LOG.warning("%s -> epochs in the gap are excluded", msg)
        elif ann.onset < covered_until - 1e-6:
            msg = (f"Overlapping annotations: {stage!r} starts at {ann.onset}s but the "
                   f"previous annotation ends at {covered_until:.1f}s")
            issues.append(msg)
            if on_conflict == "error":
                raise RecordError(f"Conflicting hypnogram annotations: {msg}")
            LOG.warning("%s -> later annotation wins on overlapping epochs", msg)

        lo = max(0, start_e)
        hi = min(n_epochs, start_e + n_e)
        if start_e + n_e > n_epochs:
            n_truncated += (start_e + n_e) - n_epochs
            issues.append(
                f"Annotation {stage!r} extends {(start_e + n_e - n_epochs)} epoch(s) beyond "
                f"the end of the EEG signal -> truncated"
            )
        if hi > lo:
            codes[lo:hi] = code
            stages[lo:hi] = stage

        table.append({
            "onset": ann.onset, "duration": ann.duration, "stage": stage,
            "code": code, "epoch_start": start_e, "n_epochs": max(0, hi - lo),
            "status": "ok" if hi > lo else "beyond_signal",
        })
        covered_until = max(covered_until, ann.onset + ann.duration)

    return AlignmentResult(codes=codes, stages=stages, ann_table=table,
                           issues=issues, n_truncated=n_truncated)


# ---------------------------------------------------------------------------
# Record pairing
# ---------------------------------------------------------------------------

@dataclass
class RecordPair:
    record_id: str
    subject_id: str
    psg_path: Path
    hyp_path: Path
    psg_header: EdfHeader
    hyp_header: EdfHeader
    pairing_method: str


def derive_subject_id(record_id: str, pattern: Optional[str]) -> Tuple[str, bool]:
    """Extract the subject id from a record id.

    Returns ``(subject_id, matched_by_regex)``. When no regex is configured or it
    does not match, the record id itself is used and a warning is emitted, so the
    split logic still groups deterministically rather than crashing.
    """
    if pattern:
        match = re.match(pattern, record_id)
        if match:
            return (match.group(1) if match.groups() else match.group(0)), True
        LOG.warning("subject_id_regex %r did not match record id %r", pattern, record_id)
        return record_id, False
    # Sleep-EDF style: SC4001E0 -> SC4001 (letters + digits, drop the trailing
    # recording index letter/digits).
    match = re.match(r"^([A-Za-z]+\d+)", record_id)
    if match:
        return match.group(1), True
    return record_id, False


def discover_pairs(input_dir: Path, psg_suffix: str, hypnogram_suffix: str,
                   subject_regex: Optional[str],
                   ) -> Tuple[List[RecordPair], List[str], List[str]]:
    """Scan ``input_dir`` for PSG/Hypnogram pairs.

    Pairing strategy, strongest evidence first:
      1. identical EDF start datetime (the authoritative recording identity),
         disambiguated by matching patient identification strings;
      2. otherwise identical file stem prefix, reported as a weaker match.
    Anything that cannot be paired is returned as an error string.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ConfigError(f"--input-dir is not a directory: {input_dir}")

    psg_files = sorted(p for p in input_dir.rglob("*") if p.is_file()
                       and p.name.lower().endswith("-psg.edf"))
    hyp_files = sorted(p for p in input_dir.rglob("*") if p.is_file()
                       and p.name.lower().endswith("hypnogram.edf"))
    if not psg_files:
        raise ConfigError(
            f"No *{psg_suffix} files found under {input_dir}. "
            f"Point --input-dir at the folder holding the Sleep-EDF EDF files."
        )
    if not hyp_files:
        raise ConfigError(f"No *{hypnogram_suffix} files found under {input_dir}")

    LOG.info("Discovered %d PSG file(s) and %d hypnogram file(s)",
             len(psg_files), len(hyp_files))

    psg_headers: Dict[Path, EdfHeader] = {}
    hyp_headers: Dict[Path, EdfHeader] = {}
    errors: List[str] = []
    for path in psg_files:
        try:
            psg_headers[path] = read_edf_header(path)
        except PipelineError as exc:
            errors.append(f"Unreadable PSG header {path.name}: {exc}")
    for path in hyp_files:
        try:
            hyp_headers[path] = read_edf_header(path)
        except PipelineError as exc:
            errors.append(f"Unreadable hypnogram header {path.name}: {exc}")

    def strip_suffix(name: str, suffix: str) -> str:
        low = name.lower()
        if low.endswith(suffix.lower()):
            return name[: len(name) - len(suffix)]
        return Path(name).stem

    pairs: List[RecordPair] = []
    used_psg: set = set()

    for hyp_path in sorted(hyp_headers):
        hyp_header = hyp_headers[hyp_path]
        hyp_stem = strip_suffix(hyp_path.name, hypnogram_suffix)

        candidates = [p for p, h in psg_headers.items()
                      if h.start_datetime == hyp_header.start_datetime
                      and p not in used_psg]
        method = "start_datetime"

        if len(candidates) > 1:
            same_patient = [p for p in candidates
                            if psg_headers[p].patient == hyp_header.patient]
            if len(same_patient) == 1:
                candidates = same_patient
            else:
                same_prefix = [p for p in candidates
                               if strip_suffix(p.name, psg_suffix)[:7] == hyp_stem[:7]]
                if len(same_prefix) == 1:
                    candidates = same_prefix
                else:
                    errors.append(
                        f"Ambiguous pairing for {hyp_path.name}: {len(candidates)} PSG files "
                        f"share start datetime {hyp_header.start_datetime} "
                        f"({[p.name for p in candidates]})"
                    )
                    continue

        if not candidates:
            # fall back to the filename prefix (Sleep-EDF: SC4001E0 vs SC4001EC)
            prefix = hyp_stem[:-1] if hyp_stem else hyp_stem
            fallback = [p for p in psg_headers
                        if p not in used_psg
                        and strip_suffix(p.name, psg_suffix).startswith(prefix)]
            if len(fallback) == 1:
                candidates = fallback
                method = "filename_prefix"
                LOG.warning(
                    "%s: no PSG with matching start datetime; falling back to filename "
                    "prefix match (%s)", hyp_path.name, candidates[0].name,
                )
            else:
                errors.append(
                    f"No PSG file could be paired with {hyp_path.name} "
                    f"(hypnogram start {hyp_header.start_datetime}, "
                    f"patient {hyp_header.patient!r}; {len(fallback)} prefix candidates)"
                )
                continue

        psg_path = candidates[0]
        psg_header = psg_headers[psg_path]
        used_psg.add(psg_path)
        record_id = strip_suffix(psg_path.name, psg_suffix)
        subject_id, _ = derive_subject_id(record_id, subject_regex)

        pairs.append(RecordPair(
            record_id=record_id,
            subject_id=subject_id,
            psg_path=psg_path,
            hyp_path=hyp_path,
            psg_header=psg_header,
            hyp_header=hyp_header,
            pairing_method=method,
        ))

    for path in psg_headers:
        if path not in used_psg:
            errors.append(f"PSG file {path.name} has no matching hypnogram")

    discovered = [str(p) for p in sorted(psg_headers)] + \
                 [str(p) for p in sorted(hyp_headers)]
    return pairs, errors, discovered


# ---------------------------------------------------------------------------
# Core per-record processing
# ---------------------------------------------------------------------------

@dataclass
class RecordResult:
    record_id: str
    subject_id: str
    ok: bool
    summary: Dict[str, Any]
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    epoch_index: Optional[np.ndarray] = None
    stage_text: Optional[List[str]] = None
    start_datetime: Optional[datetime] = None


def _load_signal(psg_path: Path, header: EdfHeader, channel: str,
                 reader: str) -> Tuple[np.ndarray, float, str]:
    """Return ``(signal_uV, native_sfreq, reader_used)``.

    Whichever backend is selected, the result is cross-checked against the other
    one so a silent reader regression cannot corrupt the dataset unnoticed.
    """
    requested = reader
    if reader == "auto":
        reader = "mne" if _mne_available() else "native"

    if reader == "mne":
        try:
            data, sfreq = read_signal_mne(psg_path, channel)
        except Exception as exc:
            if requested == "auto":
                LOG.warning("MNE reader failed on %s (%s: %s); falling back to the "
                            "native EDF reader", psg_path.name, type(exc).__name__, exc)
                idx = header.channel_index(channel)
                return read_signal_native(header, channel), float(header.sfreq[idx]), "native"
            raise RecordError(f"MNE could not read {psg_path.name}: {exc}") from exc

        try:
            reference = read_signal_native(header, channel)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Native cross-check unavailable: %s", exc)
            reference = None

        if reference is not None:
            if reference.shape != data.shape:
                raise RecordError(
                    f"Reader disagreement for {psg_path.name}: MNE returned "
                    f"{data.shape} samples, the EDF header implies {reference.shape}"
                )
            diff = float(np.max(np.abs(reference - data))) if data.size else 0.0
            if diff > 1e-3:
                LOG.warning(
                    "%s: MNE and native readers differ by up to %.6f uV "
                    "(tolerance 1e-3) - investigate before trusting the output",
                    psg_path.name, diff,
                )
            else:
                LOG.info("%s: reader cross-check OK (max |MNE - native| = %.2e uV)",
                         psg_path.name, diff)
        return data, sfreq, "mne"

    idx = header.channel_index(channel)
    data = read_signal_native(header, channel)
    return data, float(header.sfreq[idx]), "native"


def _mne_available() -> bool:
    try:
        import mne  # noqa: F401
        return True
    except Exception:
        return False


def process_record(pair: RecordPair, cfg: Dict[str, Any]) -> RecordResult:
    """Validate one PSG/hypnogram pair and build its 5-class epoch dataset."""
    rid = pair.record_id
    psg_h, hyp_h = pair.psg_header, pair.hyp_header
    epoch_duration = float(cfg["epoch_duration"])
    target_fs = float(cfg["target_fs"])
    notes: List[str] = []
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, fatal: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            notes.append(f"{name}: {detail}")
            if fatal:
                raise RecordError(f"{rid}: {name} failed - {detail}")
            LOG.warning("%s: %s failed - %s", rid, name, detail)

    # -- 1. identity ------------------------------------------------------
    check("psg_has_signal", not psg_h.is_annotation_only,
          f"PSG channels: {psg_h.labels}")
    check("hypnogram_is_annotation_file", hyp_h.is_annotation_only,
          f"hypnogram channels: {hyp_h.labels}")
    check("start_datetime_matches",
          psg_h.start_datetime == hyp_h.start_datetime,
          f"PSG {psg_h.start_datetime.isoformat()} vs hypnogram "
          f"{hyp_h.start_datetime.isoformat()}")
    check("patient_id_matches", psg_h.patient == hyp_h.patient,
          f"PSG {psg_h.patient!r} vs hypnogram {hyp_h.patient!r}", fatal=False)

    # -- 2. channel / sampling rate --------------------------------------
    ch_idx = psg_h.channel_index(cfg["channel"])
    native_fs = float(psg_h.sfreq[ch_idx])
    n_samp = int(psg_h.n_samples[ch_idx])
    check("channel_present", True,
          f"{cfg['channel']!r} -> index {ch_idx}, {n_samp} samples/record")
    check("record_duration_positive", psg_h.record_duration > 0,
          f"record duration {psg_h.record_duration}s")
    check("native_sfreq_matches_header",
          abs(native_fs - n_samp / psg_h.record_duration) < 1e-6,
          f"native {native_fs:.4f} Hz from {n_samp} samples / {psg_h.record_duration}s")

    # -- 3. signal / annotation time ranges ------------------------------
    signal_seconds = psg_h.signal_duration()
    _, annotations = read_edf_annotations(pair.hyp_path, total_duration=signal_seconds)
    ann_end = max(a.onset + a.duration for a in annotations)
    LOG.info("%s: signal %s, annotations 0 - %s (%d entries)",
             rid, _fmt_seconds(signal_seconds), _fmt_seconds(ann_end), len(annotations))
    check("annotation_within_signal", ann_end <= signal_seconds + 1e-6,
          f"annotations end at {ann_end:.1f}s, signal ends at {signal_seconds:.1f}s",
          fatal=False)
    check("annotation_starts_at_zero", abs(annotations[0].onset) < 1e-6,
          f"first annotation onset {annotations[0].onset}s", fatal=False)

    # -- 4. read the EEG signal ------------------------------------------
    signal, used_fs, reader_used = _load_signal(pair.psg_path, psg_h, cfg["channel"],
                                                cfg["reader"])
    LOG.info("%s: read %d samples at %.3f Hz via %s", rid, signal.size, used_fs, reader_used)
    if abs(used_fs - target_fs) > 1e-9:
        signal = resample_signal(signal, used_fs, target_fs)
        LOG.info("%s: resampled to %.3f Hz (%d samples)", rid, target_fs, signal.size)

    n_epoch_samples = int(round(epoch_duration * target_fs))
    n_epochs_signal = int(signal.size // n_epoch_samples)
    if n_epochs_signal == 0:
        raise RecordError(f"{rid}: signal too short for a single {epoch_duration:g}s epoch")
    remainder = int(signal.size - n_epochs_signal * n_epoch_samples)
    if remainder:
        notes.append(
            f"Trailing {remainder} sample(s) ({remainder / target_fs:.3f}s) do not fill a "
            f"complete epoch and were dropped"
        )
        LOG.warning("%s: dropping %d trailing sample(s)", rid, remainder)

    check("epoch_samples", n_epoch_samples == int(epoch_duration * target_fs),
          f"{n_epoch_samples} samples per {epoch_duration:g}s epoch at {target_fs:g} Hz")

    # -- 5. labels --------------------------------------------------------
    align = build_epoch_codes(
        annotations=annotations,
        n_epochs=n_epochs_signal,
        epoch_duration=epoch_duration,
        label_map=cfg["label_map"],
        excluded_stages=cfg["excluded_stages"],
        on_gap=cfg["on_annotation_gap"],
        on_conflict=cfg["on_stage_conflict"],
    )
    notes.extend(align.issues)

    codes = align.codes
    valid = codes != CODE_UNCOVERED
    n_uncovered = int((~valid).sum())

    # -- 6. edge trimming (TinySleepNet-compatible) -----------------------
    keep = np.zeros(n_epochs_signal, dtype=bool)
    trim_window: Dict[str, Any] = {"applied": False}
    if cfg["trim_edge_wake"]:
        edge = int(cfg["edge_wake_epochs"])
        non_wake = np.flatnonzero(valid & (codes != 0))
        if non_wake.size == 0:
            keep[:] = valid
            notes.append(
                "No non-W epoch found; edge-wake trimming skipped and every valid epoch kept"
            )
            LOG.warning("%s: no non-W epoch, edge trimming skipped", rid)
        else:
            lo = max(0, int(non_wake[0]) - edge)
            hi = min(n_epochs_signal - 1, int(non_wake[-1]) + edge)
            keep[lo:hi + 1] = True
            trim_window = {
                "applied": True,
                "start_epoch": lo,
                "end_epoch": hi,
                "first_non_wake_epoch": int(non_wake[0]),
                "last_non_wake_epoch": int(non_wake[-1]),
                "edge_wake_epochs": edge,
            }
    else:
        keep[:] = valid
        trim_window = {"applied": False, "reason": "trim_edge_wake disabled"}

    keep &= valid
    n_dropped_edge = int(valid.sum() - keep.sum())

    # -- 7. build epoch matrix -------------------------------------------
    epochs = signal[: n_epochs_signal * n_epoch_samples].reshape(n_epochs_signal,
                                                                n_epoch_samples)
    epochs = epochs.astype(np.float32, copy=False)

    n_movement = int(((codes == CODE_MOVEMENT) & keep).sum())
    n_unknown = int(((codes == CODE_UNKNOWN) & keep).sum())
    excluded_mask = keep & np.isin(codes, [CODE_MOVEMENT, CODE_UNKNOWN])

    finite = np.isfinite(epochs).all(axis=1)
    n_nonfinite = int((~finite & keep).sum())
    if n_nonfinite and not cfg["drop_nonfinite_epochs"]:
        raise RecordError(f"{rid}: {n_nonfinite} epoch(s) contain NaN/Inf")

    final_mask = keep & ~excluded_mask & finite
    if n_nonfinite:
        notes.append(f"{n_nonfinite} epoch(s) containing NaN/Inf were dropped")

    idx_final = np.flatnonzero(final_mask)
    x = np.ascontiguousarray(epochs[idx_final])
    y = codes[idx_final].astype(np.int32)

    if x.shape[0] == 0:
        raise RecordError(f"{rid}: no valid epoch left after alignment/trimming")
    if not np.isfinite(x).all():  # defensive
        raise RecordError(f"{rid}: output still contains NaN/Inf")
    if y.min() < 0 or y.max() > 4:
        raise RecordError(f"{rid}: label out of the 0-4 range ({y.min()}..{y.max()})")

    # -- 8. signal sanity -------------------------------------------------
    flat = int((np.std(x, axis=1) < 1e-9).sum())
    if flat:
        LOG.warning("%s: %d epoch(s) are perfectly flat", rid, flat)
    signal_stats = {
        "min_uV": float(x.min()), "max_uV": float(x.max()),
        "mean_uV": float(x.mean()), "std_uV": float(x.std()),
        "flat_epochs": flat,
    }
    if signal_stats["min_uV"] < -1e4 or signal_stats["max_uV"] > 1e4:
        notes.append(
            f"Amplitude range {signal_stats['min_uV']:.1f}..{signal_stats['max_uV']:.1f} uV "
            f"is unusually large for EEG; check the channel/units"
        )

    # -- 9. accounting ----------------------------------------------------
    label_counts = {LABEL_NAMES[k]: int((y == k).sum()) for k in range(5)}
    accounting = {
        "n_epochs_in_signal": n_epochs_signal,
        "n_epochs_uncovered_by_annotation": n_uncovered,
        "n_epochs_annotation_beyond_signal": int(align.n_truncated),
        "n_epochs_eligible_before_trim": int(valid.sum()),
        "edge_trim": trim_window,
        "n_dropped_by_edge_trim": n_dropped_edge,
        "n_dropped_movement": n_movement,
        "n_dropped_unknown_stage": n_unknown,
        "n_dropped_nonfinite": n_nonfinite,
        "n_epochs_final": int(x.shape[0]),
        "n_epochs_final_check_ok": bool(
            int(x.shape[0]) == int(keep.sum()) - n_movement - n_unknown - n_nonfinite
        ),
    }
    if not accounting["n_epochs_final_check_ok"]:
        raise RecordError(f"{rid}: epoch accounting does not balance")

    raw_stage_percent = {
        stage: round(100.0 * n_epochs / n_epochs_signal, 2)
        for stage, n_epochs in sorted(
            {s: int((align.stages == s).sum()) for s in set(align.stages) if s}.items()
        )
    }

    summary = {
        "record_id": rid,
        "subject_id": pair.subject_id,
        "status": "ok",
        "psg_file": pair.psg_path.name,
        "hypnogram_file": pair.hyp_path.name,
        "pairing_method": pair.pairing_method,
        "psg_sha256": sha256_file(pair.psg_path),
        "hypnogram_sha256": sha256_file(pair.hyp_path),
        "start_datetime": psg_h.start_datetime.isoformat(),
        "patient": psg_h.patient,
        "signal_duration_s": round(signal_seconds, 3),
        "annotation_duration_s": round(ann_end, 3),
        "n_annotation_entries": len(annotations),
        "channel": cfg["channel"],
        "native_sfreq": round(native_fs, 4),
        "target_fs": target_fs,
        "reader": reader_used,
        "epoch_duration_s": epoch_duration,
        "n_epoch_samples": n_epoch_samples,
        "npz_file": f"{rid}.npz",
        "epoch_accounting": accounting,
        "label_distribution": label_counts,
        "label_percent": {k: round(100.0 * v / x.shape[0], 2) for k, v in label_counts.items()},
        "raw_stage_share_of_record_percent": raw_stage_percent,
        "signal_stats": signal_stats,
        "checks": checks,
        "notes": notes,
    }

    return RecordResult(
        record_id=rid, subject_id=pair.subject_id, ok=True, summary=summary,
        x=x, y=y, epoch_index=idx_final.astype(np.int32),
        stage_text=[str(s) for s in align.stages[idx_final]],
        start_datetime=psg_h.start_datetime,
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def save_npz_deterministic(path: Path, arrays: Dict[str, Any]) -> None:
    """Write an ``.npz`` with fixed zip timestamps so reruns are byte-identical."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[key]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buffer.getvalue())
    os.replace(tmp, path)


def build_npz_payload(res: RecordResult, cfg: Dict[str, Any]) -> Dict[str, Any]:
    assert res.x is not None and res.y is not None
    acc = res.summary["epoch_accounting"]
    return {
        # --- the 7 keys required by this task ---
        "x": res.x.astype(np.float32, copy=False),
        "y": res.y.astype(np.int32, copy=False),
        "fs": np.int32(int(cfg["target_fs"])),
        "ch_label": np.array(str(cfg["channel"])),
        "epoch_duration": np.int32(int(cfg["epoch_duration"])),
        "record_id": np.array(str(res.record_id)),
        "subject_id": np.array(str(res.subject_id)),
        # --- extras that make the file self-describing ---
        "epoch_index": res.epoch_index.astype(np.int32),
        "label_names": np.array([LABEL_NAMES[k] for k in range(5)]),
        "start_datetime": np.array(res.start_datetime.isoformat()),  # type: ignore[union-attr]
        "preprocessing_version": np.array(__version__),
        # --- keys the reference TinySleepNet npz also carries, so existing
        #     training/eval code can read this file unchanged ---
        "n_epochs": np.int32(acc["n_epochs_final"]),
        "n_all_epochs": np.int32(acc["n_epochs_in_signal"]),
        "file_duration": np.float64(res.summary["signal_duration_s"]),
    }


def write_epoch_metadata(path: Path, results: Sequence[RecordResult],
                         epoch_duration: float) -> int:
    columns = [
        "record_id", "subject_id", "epoch_seq", "epoch_index", "start_time_sec",
        "start_datetime", "label", "label_name", "raw_stage",
    ]
    epoch_sec = int(round(epoch_duration))
    n_rows = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for res in results:
            if not res.ok or res.y is None or res.epoch_index is None:
                continue
            assert res.start_datetime is not None
            for seq, (orig, label) in enumerate(zip(res.epoch_index, res.y)):
                start_sec = int(orig) * epoch_sec
                writer.writerow([
                    res.record_id, res.subject_id, seq, int(orig), start_sec,
                    (res.start_datetime + timedelta(seconds=start_sec)).isoformat(
                        sep=" ", timespec="seconds"),
                    int(label), LABEL_NAMES[int(label)],
                    (res.stage_text[seq] if res.stage_text else ""),
                ])
                n_rows += 1
    return n_rows


# ---------------------------------------------------------------------------
# Subject-level split
# ---------------------------------------------------------------------------

def build_subject_split(records: Sequence[Dict[str, Any]],
                        seed: int = 42,
                        ratios: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Deterministic subject-level train/val/test split.

    Rules: one subject appears in exactly one split; epochs are never shuffled
    across splits; the split is refused (``feasible: false``) rather than faked
    when there are not enough subjects.
    """
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    by_subject: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        sub = str(rec["subject_id"])
        entry = by_subject.setdefault(sub, {"records": [], "n_epochs": 0,
                                            "label_distribution": {n: 0 for n in LABEL_NAMES.values()}})
        entry["records"].append(rec["record_id"])
        n = int(rec["epoch_accounting"]["n_epochs_final"])
        entry["n_epochs"] += n
        for name, cnt in rec["label_distribution"].items():
            entry["label_distribution"][name] += int(cnt)

    subjects = sorted(by_subject)
    n_subjects = len(subjects)

    base = {
        "seed": seed,
        "ratios": ratios,
        "n_subjects": n_subjects,
        "n_records": sum(len(v["records"]) for v in by_subject.values()),
        "n_epochs": sum(v["n_epochs"] for v in by_subject.values()),
    }

    n_test = max(1, int(round(ratios.get("test", 0.15) * n_subjects)))
    n_val = max(1, int(round(ratios.get("val", 0.15) * n_subjects)))
    n_train = n_subjects - n_val - n_test

    if n_subjects < 3 or n_train < 1:
        reason = (
            f"Only {n_subjects} subject(s) available; a 70/15/15 subject-level split "
            f"needs at least 3 subjects (ideally >= 20). No split was produced."
        )
        LOG.warning("Subject split NOT created: %s", reason)
        return {**base, "feasible": False, "reason": reason,
                "subjects": {"train": [], "val": [], "test": []},
                "records": {"train": [], "val": [], "test": []},
                "stats": {}, "checks": {"disjoint": None, "covers_all": None}}

    rng = np.random.default_rng(seed)
    order = [subjects[i] for i in rng.permutation(n_subjects)]
    splits = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }

    stats: Dict[str, Any] = {}
    for name, subs in splits.items():
        n_ep = sum(by_subject[s]["n_epochs"] for s in subs)
        dist = {k: 0 for k in LABEL_NAMES.values()}
        for s in subs:
            for k, v in by_subject[s]["label_distribution"].items():
                dist[k] += v
        stats[name] = {
            "n_subjects": len(subs),
            "n_records": sum(len(by_subject[s]["records"]) for s in subs),
            "n_epochs": n_ep,
            "epoch_share": round(n_ep / max(1, base["n_epochs"]), 4),
            "subject_share": round(len(subs) / n_subjects, 4),
            "label_distribution": dist,
        }

    sets = [set(splits[k]) for k in ("train", "val", "test")]
    disjoint = not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    covers_all = set().union(*sets) == set(subjects)
    if not (disjoint and covers_all):
        raise RecordError(f"Subject split self-check failed (disjoint={disjoint}, "
                          f"covers_all={covers_all})")

    LOG.info("Subject split: train=%d val=%d test=%d subjects (seed=%d)",
             len(splits["train"]), len(splits["val"]), len(splits["test"]), seed)

    return {
        **base,
        "feasible": True,
        "reason": "ok",
        "subjects": splits,
        "records": {k: sorted(r for s in v for r in by_subject[s]["records"])
                    for k, v in splits.items()},
        "per_subject": {s: {"records": by_subject[s]["records"],
                            "n_epochs": by_subject[s]["n_epochs"]} for s in subjects},
        "stats": stats,
        "checks": {"disjoint": disjoint, "covers_all": covers_all,
                   "no_epoch_shuffling": True},
    }


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def load_config_file(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "PyYAML is required to read a .yaml config; install it or use a "
                ".json config file instead"
            ) from exc
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"Unsupported config extension {path.suffix!r} (use .yaml/.yml/.json)")
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must contain a mapping at the top level")
    return data


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge defaults <- config file <- explicit CLI flags (CLI wins)."""
    cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULTS))  # deep copy
    given = lambda key: getattr(args, key, None)  # noqa: E731  (SUPPRESS -> absent)

    if given("config"):
        file_cfg = load_config_file(Path(args.config))
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            LOG.warning("Ignoring unknown config keys: %s", sorted(unknown))
        for key, value in file_cfg.items():
            if key in DEFAULTS:
                cfg[key] = value

    for key in ("input_dir", "output_dir", "psg", "hypnogram", "channel",
                "target_fs", "epoch_duration", "edge_wake_epochs", "reader",
                "log_level", "subject_id_regex", "split_seed"):
        value = given(key)
        if value is not None:
            cfg[key] = value
    if given("no_trim_edge_wake"):
        cfg["trim_edge_wake"] = False
    if given("overwrite"):
        cfg["overwrite"] = True
    if given("strict"):
        cfg["strict"] = True
        cfg["on_annotation_gap"] = "error"
    if given("make_split"):
        cfg["make_split"] = True

    # normalise / validate
    if not cfg.get("psg_suffix"):
        cfg["psg_suffix"] = DEFAULTS["psg_suffix"]
    if not cfg.get("hypnogram_suffix"):
        cfg["hypnogram_suffix"] = DEFAULTS["hypnogram_suffix"]

    # ---- resolve defaults so a bare "Run" in an IDE works -----------------
    # Track which values we invented, so the log can say so plainly instead of
    # pretending the user asked for them.
    cfg.setdefault("_resolved", {})
    resolved: Dict[str, str] = cfg["_resolved"]

    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(project_root() / DEFAULT_OUTPUT_RELPATH)
        resolved["output_dir"] = "default"

    if cfg["input_dir"] is None and not (cfg["psg"] and cfg["hypnogram"]):
        found, searched = find_default_input_dir()
        if found is None:
            raise UsageError(
                "Could not find any *-PSG.edf recording, and no --input-dir / "
                "--psg was given.",
                hint=_usage_hint(searched),
                searched=[str(p) for p in searched],
            )
        cfg["input_dir"] = str(found)
        resolved["input_dir"] = "auto-discovered"

    if cfg["input_dir"] and (cfg["psg"] or cfg["hypnogram"]):
        if not (cfg["psg"] and cfg["hypnogram"]):
            raise ConfigError(
                "--psg and --hypnogram must be given together (both are needed to "
                "pair a recording with its scoring)"
            )
        raise ConfigError(
            "Use either --input-dir (scan a folder) or --psg/--hypnogram "
            "(one explicit pair) - not both"
        )

    epoch_duration = float(cfg["epoch_duration"])
    if epoch_duration <= 0:
        raise ConfigError("epoch_duration must be positive")
    if int(round(epoch_duration)) != epoch_duration:
        raise ConfigError("epoch_duration must be an integer number of seconds")
    if int(cfg["target_fs"]) <= 0:
        raise ConfigError("target_fs must be positive")
    if int(cfg["edge_wake_epochs"]) < 0:
        raise ConfigError("edge_wake_epochs must be >= 0")
    if cfg["reader"] not in ("auto", "mne", "native"):
        raise ConfigError("reader must be one of: auto, mne, native")
    if cfg["on_annotation_gap"] not in ("drop", "error"):
        raise ConfigError("on_annotation_gap must be 'drop' or 'error'")
    if cfg["on_stage_conflict"] not in ("drop", "error"):
        raise ConfigError("on_stage_conflict must be 'drop' or 'error'")

    ratios = cfg["split_ratios"]
    total = sum(float(ratios.get(k, 0)) for k in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"split_ratios must sum to 1.0 (got {total})")

    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI flags.

    Every optional flag uses ``SUPPRESS`` so an omitted flag is simply absent from
    the namespace; that is how we tell "not given on the command line" apart from
    "explicitly set to the default value", which lets config-file values survive.
    """
    parser = argparse.ArgumentParser(
        description="Build 30 s / 5-class Sleep-EDF epochs from paired EDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "runs with NO arguments:\n"
            "  both folders are resolved for you. --output-dir defaults to\n"
            "  <project>/outputs/prepared, and the EDF folder is auto-detected from\n"
            f"  ${ENV_DATA_DIR} or a conventional data/ folder. If nothing is found\n"
            "  you get a list of the folders that were searched.\n"
            "\n"
            "examples:\n"
            "  just run it (auto-detect everything):\n"
            "    python prepare_sleepedf.py\n"
            "  point it at a folder of recordings:\n"
            "    python prepare_sleepedf.py --input-dir ./sleep-cassette\n"
            "  one explicit pair:\n"
            "    python prepare_sleepedf.py --psg SC4001E0-PSG.edf \\\n"
            "        --hypnogram SC4001EC-Hypnogram.edf\n"
            "  folder + subject split + custom output:\n"
            "    python prepare_sleepedf.py --input-dir ./sleep-cassette \\\n"
            "        --output-dir ./out --make-split\n"
            "  with a config file (CLI flags still win):\n"
            "    python prepare_sleepedf.py --config config.yaml\n"
            "\n"
            "exit codes: 0 all records ok, 1 some record/pairing failed, "
            "2 configuration/usage problem\n"
        ),
    )
    io_group = parser.add_argument_group("input / output")
    io_group.add_argument("--input-dir", default=argparse.SUPPRESS,
                          help="Directory scanned recursively for *-PSG.edf / "
                               "*-Hypnogram.edf [default: auto-detected]")
    io_group.add_argument("--output-dir", default=argparse.SUPPRESS,
                          help="Where .npz + manifest + csv + log are written "
                               f"[default: <project>/{DEFAULT_OUTPUT_RELPATH.as_posix()}]")
    io_group.add_argument("--psg", default=argparse.SUPPRESS,
                          help="Single PSG EDF file (must be used with --hypnogram)")
    io_group.add_argument("--hypnogram", default=argparse.SUPPRESS,
                          help="Single hypnogram EDF file (must be used with --psg)")
    io_group.add_argument("--config", default=argparse.SUPPRESS,
                          help="YAML or JSON config file; explicit CLI flags win over it")
    io_group.add_argument("--overwrite", action="store_true", default=argparse.SUPPRESS,
                          help="Allow replacing files this tool already wrote "
                               "[default: off, and nothing is ever deleted]")

    sig_group = parser.add_argument_group("signal / epoch")
    sig_group.add_argument("--channel", default=argparse.SUPPRESS,
                          help='EEG channel name [default: "EEG Fpz-Cz"]')
    sig_group.add_argument("--target-fs", type=float, default=argparse.SUPPRESS,
                          help="Target sampling rate in Hz [default: 100]")
    sig_group.add_argument("--epoch-duration", type=float, default=argparse.SUPPRESS,
                          help="Epoch length in seconds (integer) [default: 30]")
    sig_group.add_argument("--reader", choices=["auto", "mne", "native"],
                          default=argparse.SUPPRESS,
                          help="EDF signal reader backend [default: auto]")
    sig_group.add_argument("--no-trim-edge-wake", action="store_true",
                          default=argparse.SUPPRESS,
                          help="Keep every scored epoch instead of trimming long wake "
                               "runs at both ends [default: trimming on]")
    sig_group.add_argument("--edge-wake-epochs", type=int, default=argparse.SUPPRESS,
                          help="Wake epochs kept before/after the scored sleep period "
                               "[default: 60, i.e. 30 min]")

    split_group = parser.add_argument_group("subject split")
    split_group.add_argument("--make-split", action="store_true", default=argparse.SUPPRESS,
                             help="Also write subject_split.json")
    split_group.add_argument("--split-seed", type=int, default=argparse.SUPPRESS,
                             help="Random seed for the subject split [default: 42]")
    split_group.add_argument("--subject-id-regex", default=argparse.SUPPRESS,
                             help=r"Regex with one capture group, e.g. '^(SC\d{4})' "
                                  r"[default: SC4001E0 -> SC4001]")

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument("--strict", action="store_true", default=argparse.SUPPRESS,
                            help="Treat annotation gaps/overlaps as fatal errors")
    misc_group.add_argument("--log-level", default=argparse.SUPPRESS,
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="Console log level [default: INFO; the log file is "
                                 "always DEBUG]")
    misc_group.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging / output directory guard
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("prepare_sleepedf")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logging.getLogger("mne").setLevel(logging.WARNING)
    return logger


def teardown_logging() -> None:
    """Flush and close every handler.

    Windows keeps an open file locked, so a handler that is left attached to the
    logger would block the output directory from being moved or deleted by the
    caller (and by tests using TemporaryDirectory).
    """
    logger = logging.getLogger("prepare_sleepedf")
    for handler in list(logger.handlers):
        try:
            handler.flush()
        except Exception:  # pragma: no cover - already closed
            pass
        try:
            handler.close()
        finally:
            logger.removeHandler(handler)


def guard_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output dir but never delete anything that is already there."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(f"--output-dir exists and is not a directory: {output_dir}")

    if output_dir.exists():
        existing = sorted(p.name for p in output_dir.iterdir())
        owned = ("dataset_manifest.json", "epoch_metadata.csv", "preprocessing.log",
                 "subject_split.json")
        has_owned = any(name in existing for name in owned) or any(
            name.endswith(".npz") for name in existing)
        if existing and has_owned and not overwrite:
            raise ConfigError(
                f"Output directory already has results from a previous run:\n"
                f"  {output_dir}\n"
                f"  existing entries: {existing[:8]}{' ...' if len(existing) > 8 else ''}\n"
                f"\n"
                f"Nothing was deleted or overwritten. Pick one:\n"
                f"  - keep this run's result and write elsewhere:\n"
                f"        --output-dir another_folder\n"
                f"  - or replace it (only this tool's own files are rewritten):\n"
                f"        --overwrite"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_pairs(cfg: Dict[str, Any]) -> Tuple[List[RecordPair], List[str], List[str]]:
    if cfg["psg"]:
        psg_path = Path(cfg["psg"]).expanduser()
        hyp_path = Path(cfg["hypnogram"]).expanduser()
        for p in (psg_path, hyp_path):
            if not p.is_file():
                raise ConfigError(f"Input file not found: {p}")
        psg_header = read_edf_header(psg_path)
        hyp_header = read_edf_header(hyp_path)
        record_id = psg_path.name
        suffix = cfg["psg_suffix"] or ""
        if suffix and record_id.lower().endswith(suffix.lower()):
            record_id = record_id[: -len(suffix)]
        else:
            record_id = psg_path.stem
        subject_id, _ = derive_subject_id(record_id, cfg["subject_id_regex"])
        pair = RecordPair(record_id=record_id, subject_id=subject_id,
                          psg_path=psg_path, hyp_path=hyp_path,
                          psg_header=psg_header, hyp_header=hyp_header,
                          pairing_method="explicit_files")
        return [pair], [], [str(psg_path), str(hyp_path)]

    return discover_pairs(Path(cfg["input_dir"]), cfg["psg_suffix"],
                          cfg["hypnogram_suffix"], cfg["subject_id_regex"])


def run(cfg: Dict[str, Any]) -> int:
    """Public entry point: runs the pipeline and always releases the log file."""
    try:
        return _run(cfg)
    finally:
        teardown_logging()


def _run(cfg: Dict[str, Any]) -> int:
    output_dir = Path(cfg["output_dir"]).expanduser()

    # Never let the writer touch the folder that holds the source recordings.
    input_dir = Path(cfg["input_dir"]).expanduser() if cfg["input_dir"] else None
    nested_output = False
    if input_dir is not None:
        try:
            if output_dir.resolve() == input_dir.resolve():
                raise ConfigError(
                    f"--output-dir must not be the same directory as --input-dir "
                    f"({input_dir}); the raw EDF files are read-only inputs."
                )
            nested_output = input_dir.resolve() in output_dir.resolve().parents
        except OSError:  # pragma: no cover - unresolved path on odd filesystems
            pass

    guard_output_dir(output_dir, bool(cfg["overwrite"]))

    log_path = output_dir / "preprocessing.log"
    logger = setup_logging(log_path, cfg["log_level"])
    if nested_output:
        logger.warning(
            "--output-dir sits inside --input-dir; only .npz/.json/.csv/.log files are "
            "written there, the EDF inputs are never opened for writing."
        )
    logger.info("=" * 78)
    logger.info("prepare_sleepedf %s - Sleep-EDF preprocessing", __version__)
    logger.info("started: %s", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    logger.info("python: %s | platform: %s", platform.python_version(), platform.platform())
    logger.info("numpy: %s | mne: %s", np.__version__,
                _module_version("mne") or "not installed")

    # Say plainly where each path came from, so an auto-resolved run is never
    # mistaken for the tool silently reading an unexpected folder.
    resolved = cfg.get("_resolved") or {}
    if input_dir is not None:
        origin = resolved.get("input_dir")
        if origin == "auto-discovered":
            logger.info("input dir : %s  (auto-discovered; pass --input-dir to override)",
                        input_dir)
        else:
            logger.info("input dir : %s", input_dir)
    else:
        logger.info("input     : explicit --psg / --hypnogram pair")

    if resolved.get("output_dir") == "default":
        logger.info("output dir: %s  (default; pass --output-dir to override)",
                    output_dir)
    else:
        logger.info("output dir: %s", output_dir)

    pairs, pairing_errors, discovered = collect_pairs(cfg)
    logger.info("pairs to process: %d", len(pairs))
    for err in pairing_errors:
        logger.error("PAIRING: %s", err)

    if not pairs:
        logger.error("No usable PSG/hypnogram pair was found - aborting")
        write_json_atomic(output_dir / "dataset_manifest.json", {
            "schema_version": "1.0",
            "tool": {"name": "prepare_sleepedf", "version": __version__},
            "errors": pairing_errors,
            "records": [],
        })
        return 1

    results: List[RecordResult] = []
    record_errors: List[Dict[str, str]] = list(
        {"record_id": "<pairing>", "error": e} for e in pairing_errors
    )

    for pair in pairs:
        logger.info("-" * 78)
        logger.info("record %s (subject %s)", pair.record_id, pair.subject_id)
        logger.info("  PSG       : %s", pair.psg_path)
        logger.info("  hypnogram : %s", pair.hyp_path)
        logger.info("  paired by : %s", pair.pairing_method)
        try:
            res = process_record(pair, cfg)
        except PipelineError as exc:
            logger.error("%s FAILED: %s", pair.record_id, exc)
            record_errors.append({"record_id": pair.record_id, "error": str(exc)})
            if cfg["strict"]:
                raise
            results.append(RecordResult(
                record_id=pair.record_id, subject_id=pair.subject_id, ok=False,
                summary={"record_id": pair.record_id, "subject_id": pair.subject_id,
                         "status": "failed", "error": str(exc)},
            ))
            continue
        except Exception as exc:  # unexpected: keep the batch alive, but be loud
            logger.exception("%s CRASHED: %s", pair.record_id, exc)
            record_errors.append({"record_id": pair.record_id,
                                  "error": f"unexpected {type(exc).__name__}: {exc}"})
            results.append(RecordResult(
                record_id=pair.record_id, subject_id=pair.subject_id, ok=False,
                summary={"record_id": pair.record_id, "subject_id": pair.subject_id,
                         "status": "failed", "error": str(exc)},
            ))
            continue

        npz_path = output_dir / f"{res.record_id}.npz"
        save_npz_deterministic(npz_path, build_npz_payload(res, cfg))
        logger.info("  wrote %s", npz_path.name)
        acc = res.summary["epoch_accounting"]
        logger.info(
            "  epochs: signal=%d eligible=%d -> edge_trim=-%d movement=-%d unknown=-%d "
            "nonfinite=-%d => final=%d",
            acc["n_epochs_in_signal"], acc["n_epochs_eligible_before_trim"],
            acc["n_dropped_by_edge_trim"], acc["n_dropped_movement"],
            acc["n_dropped_unknown_stage"], acc["n_dropped_nonfinite"],
            acc["n_epochs_final"],
        )
        logger.info("  label distribution: %s", res.summary["label_distribution"])
        results.append(res)

    ok_results = [r for r in results if r.ok]

    # --- metadata csv ---------------------------------------------------
    csv_path = output_dir / "epoch_metadata.csv"
    n_rows = write_epoch_metadata(csv_path, ok_results, float(cfg["epoch_duration"]))
    logger.info("wrote %s (%d rows)", csv_path.name, n_rows)

    # --- subject split --------------------------------------------------
    split_payload: Optional[Dict[str, Any]] = None
    if cfg["make_split"]:
        split_payload = build_subject_split(
            [r.summary for r in ok_results],
            seed=int(cfg["split_seed"]),
            ratios=cfg["split_ratios"],
        )
        write_json_atomic(output_dir / "subject_split.json", split_payload)

    # --- manifest -------------------------------------------------------
    totals_dist = {name: 0 for name in LABEL_NAMES.values()}
    for res in ok_results:
        for name, cnt in res.summary["label_distribution"].items():
            totals_dist[name] += int(cnt)
    n_total = sum(totals_dist.values())

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "name": "prepare_sleepedf",
            "version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mne": _module_version("mne"),
            "reference": "https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py",
        },
        "dataset": {
            "name": "Sleep-EDF Expanded (Sleep Cassette)",
            "input_dir": cfg["input_dir"] or "<explicit files>",
            "output_dir": str(output_dir),
            # "default" / "auto-discovered" when the value was inferred rather
            # than passed on the command line, so runs stay explainable.
            "path_origin": dict(cfg.get("_resolved") or {}),
            "discovered_files": discovered,
            "n_records_ok": len(ok_results),
            "n_records_failed": len(record_errors),
            "n_subjects": len({r.subject_id for r in ok_results}),
        },
        "processing_parameters": {
            "channel": cfg["channel"],
            "target_fs": cfg["target_fs"],
            "epoch_duration": cfg["epoch_duration"],
            "epoch_samples": int(round(float(cfg["epoch_duration"]) * float(cfg["target_fs"]))),
            "label_map": cfg["label_map"],
            "excluded_stages": cfg["excluded_stages"],
            "label_names": LABEL_NAMES,
            "trim_edge_wake": cfg["trim_edge_wake"],
            "edge_wake_epochs": cfg["edge_wake_epochs"],
            "on_annotation_gap": cfg["on_annotation_gap"],
            "on_stage_conflict": cfg["on_stage_conflict"],
            "drop_nonfinite_epochs": cfg["drop_nonfinite_epochs"],
            "reader": cfg["reader"],
            "signal_transform": "none (raw values in microvolts; float32)",
            "alignment_rule": (
                "Epoch grid anchored at the first PSG sample; epoch k covers "
                "[k*epoch_duration, (k+1)*epoch_duration). An annotation labels the "
                "epochs fully covered by [onset, onset+duration) and is accepted only "
                "when both onset and duration are exact multiples of epoch_duration."
            ),
            "preprocessing_is_deterministic": True,
        },
        "pairing": [
            {
                "record_id": r.record_id,
                "subject_id": r.subject_id,
                "psg_file": r.summary.get("psg_file"),
                "hypnogram_file": r.summary.get("hypnogram_file"),
                "pairing_method": r.summary.get("pairing_method"),
                "psg_sha256": r.summary.get("psg_sha256"),
                "hypnogram_sha256": r.summary.get("hypnogram_sha256"),
                "start_datetime": r.summary.get("start_datetime"),
            }
            for r in ok_results
        ],
        "records": [r.summary for r in results],
        "totals": {
            "n_epochs": n_total,
            "label_distribution": totals_dist,
            "label_percent": {k: round(100.0 * v / n_total, 2) if n_total else 0.0
                              for k, v in totals_dist.items()},
        },
        "exclusions": {
            "movement_time_epochs": sum(r.summary["epoch_accounting"]["n_dropped_movement"]
                                        for r in ok_results),
            "unknown_stage_epochs": sum(r.summary["epoch_accounting"]["n_dropped_unknown_stage"]
                                        for r in ok_results),
            "edge_trim_epochs": sum(r.summary["epoch_accounting"]["n_dropped_by_edge_trim"]
                                    for r in ok_results),
            "uncovered_epochs": sum(r.summary["epoch_accounting"]["n_epochs_uncovered_by_annotation"]
                                    for r in ok_results),
            "nonfinite_epochs": sum(r.summary["epoch_accounting"]["n_dropped_nonfinite"]
                                    for r in ok_results),
        },
        "anomalies": [
            {"record_id": r.record_id, "notes": r.summary.get("notes", [])}
            for r in ok_results if r.summary.get("notes")
        ],
        "errors": record_errors,
        "subject_split": split_payload,
        # the manifest itself is not on disk yet when this is computed
        "outputs": sorted({p.name for p in output_dir.iterdir()} | {"dataset_manifest.json"}),
        "no_split_reason": (
            None if (split_payload and split_payload.get("feasible"))
            else (split_payload or {}).get("reason") if split_payload
            else "subject split not requested (pass --make-split)"
        ),
    }
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)

    logger.info("-" * 78)
    logger.info("TOTAL epochs: %d | distribution: %s", n_total, totals_dist)
    if n_total:
        logger.info("TOTAL label percent: %s",
                    {k: f"{100.0 * v / n_total:.2f}%" for k, v in totals_dist.items()})
    logger.info("exclusions: %s", manifest["exclusions"])
    if record_errors:
        logger.error("%d record(s)/pairing step(s) failed - see dataset_manifest.json",
                     len(record_errors))
    logger.info("wrote %s", output_dir / "dataset_manifest.json")
    logger.info("done")

    return 0 if not record_errors else 1


def _module_version(name: str) -> Optional[str]:
    try:
        return getattr(__import__(name), "__version__", "unknown")
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args(argv)
    try:
        cfg = build_config(args)
        return run(cfg)
    except UsageError as exc:
        # Incomplete invocation (most often: pressed Run in an IDE with no
        # arguments). Print guidance, not a bare one-liner.
        print("=" * 72, file=sys.stderr)
        print("CANNOT START: input data not found", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        if exc.hint:
            print(exc.hint, file=sys.stderr)
        print("", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print("Tip: run with --help to see every option.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        print("Tip: run with --help to see every option.", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
