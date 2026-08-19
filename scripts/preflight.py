#!/usr/bin/env python
"""Pre-run validation: check everything a run needs before a single token is
generated, and report EVERY problem found rather than stopping at the first.

The experiment runs unattended for hours on a remote box. A wrong path that only
surfaces in the final report, after three hours of generation, is the worst
possible outcome, so this is deliberately cheap: it loads no model and reads no
image pixels.

Run: python scripts/preflight.py --config CFG [CFG ...] --limit N \\
         --annotations A.jsonl --vocabulary V.json [--questions Q.jsonl] \\
         [--direction-terms D.json] [--arms NAME ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pyprojroot import here

try:
    from vr_modality_bias.data.annotations import (
        read_object_annotations,
        read_question_annotations,
    )
    from vr_modality_bias.data.manifests import read_manifest
    from vr_modality_bias.data.vocabulary import load_vocabulary
    from vr_modality_bias.utils.config import load_config
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import (
        read_object_annotations,
        read_question_annotations,
    )
    from src.vr_modality_bias.data.manifests import read_manifest
    from src.vr_modality_bias.data.vocabulary import load_vocabulary
    from src.vr_modality_bias.utils.config import load_config


KNOWN_ARMS: tuple[str, ...] = (
    "baseline",
    "arm1_sparc",
    "arm2_adaptive",
    "arm3_qcond",
    "arm4_conserve",
    "arm5_reflayer",
)

_SAMPLE = 5


@dataclass
class Findings:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "Findings") -> "Findings":
        self.problems.extend(other.problems)
        self.notes.extend(other.notes)
        return self

    @property
    def ok(self) -> bool:
        return not self.problems


def check_configs(paths: list[Path]) -> tuple[Findings, list[dict]]:
    findings = Findings()
    configs: list[dict] = []
    for path in paths:
        if not path.is_file():
            findings.problems.append(f"config not found: {path}")
            continue
        try:
            cfg = load_config(path)
        except Exception as exc:
            findings.problems.append(f"config unreadable: {path}: {exc}")
            continue
        for key in ("dataset", "model", "generation", "run"):
            if key not in cfg:
                findings.problems.append(f"config {path} has no {key!r} block")
        configs.append(cfg)
        findings.notes.append(f"config OK: {path}")
    return findings, configs


def check_manifest(cfg: dict, limit: int) -> tuple[Findings, list]:
    findings = Findings()
    try:
        manifest_path = Path(cfg["dataset"]["manifest_path"])
    except KeyError:
        findings.problems.append("config has no dataset.manifest_path")
        return findings, []

    if not manifest_path.is_file():
        findings.problems.append(f"manifest not found: {manifest_path}")
        return findings, []
    try:
        records = read_manifest(manifest_path)
    except Exception as exc:
        findings.problems.append(f"manifest unreadable: {manifest_path}: {exc}")
        return findings, []

    if len(records) < limit:
        findings.problems.append(
            f"manifest {manifest_path} holds {len(records)} item(s) but the run "
            f"asks for {limit}"
        )
    else:
        findings.notes.append(
            f"manifest OK: {manifest_path} ({len(records)} item(s), using {limit})"
        )
    return findings, records[:limit]


def check_images(cfg: dict, records: list) -> Findings:
    findings = Findings()
    try:
        images_dir = Path(cfg["dataset"]["images_dir"])
    except KeyError:
        findings.problems.append("config has no dataset.images_dir")
        return findings
    if not images_dir.is_dir():
        findings.problems.append(f"images_dir not found: {images_dir}")
        return findings

    missing = [
        (r.image_id, images_dir / r.file_name)
        for r in records
        if not (images_dir / r.file_name).is_file()
    ]
    if missing:
        findings.problems.append(
            f"{len(missing)} of {len(records)} image file(s) listed in the "
            f"manifest are not on disk, e.g. "
            + "; ".join(f"{i}->{p}" for i, p in missing[:_SAMPLE])
        )
    else:
        findings.notes.append(f"images OK: {len(records)} file(s) present under {images_dir}")
    return findings


def check_object_annotations(path: Path) -> tuple[Findings, set[str]]:
    findings = Findings()
    if not path.is_file():
        findings.problems.append(f"annotations not found: {path}")
        return findings, set()
    try:
        records = read_object_annotations(path)
    except Exception as exc:
        findings.problems.append(f"annotations unreadable: {path}: {exc}")
        return findings, set()
    findings.notes.append(f"annotations OK: {path} ({len(records)} item(s))")
    return findings, {r.image_id for r in records}


def check_question_annotations(path: Path) -> tuple[Findings, set[str]]:
    findings = Findings()
    if not path.is_file():
        findings.problems.append(f"questions not found: {path}")
        return findings, set()
    try:
        records = read_question_annotations(path)
    except Exception as exc:
        findings.problems.append(f"questions unreadable: {path}: {exc}")
        return findings, set()
    findings.notes.append(
        f"questions OK: {path} ({len(records)} question(s) over "
        f"{len({r.image_id for r in records})} image(s))"
    )
    return findings, {r.image_id for r in records}


def check_ids_cross(
    manifest_ids: set[str], annotation_ids: set[str], *, what: str
) -> Findings:
    """The single most valuable check here.

    If the two id spaces do not meet, generation runs to completion and the
    report is either empty or fatal at the very end. Reporting the overlap and
    a sample of each side turns a three-hour mystery into a five-second fix.
    """
    findings = Findings()
    if not manifest_ids or not annotation_ids:
        findings.problems.append(
            f"cannot cross ids with {what}: one of the two sides is empty "
            f"(manifest={len(manifest_ids)}, {what}={len(annotation_ids)})"
        )
        return findings

    matched = manifest_ids & annotation_ids
    only_manifest = sorted(manifest_ids - annotation_ids)
    only_annotation = sorted(annotation_ids - manifest_ids)

    if not matched:
        findings.problems.append(
            f"NO id in the manifest appears in {what}. The two sides use "
            f"different formats, so the whole run would produce nothing.\n"
            f"      manifest sample : {sorted(manifest_ids)[:_SAMPLE]}\n"
            f"      {what} sample : {sorted(annotation_ids)[:_SAMPLE]}"
        )
        return findings

    findings.notes.append(
        f"id crossing with {what}: {len(matched)} matched, "
        f"{len(only_manifest)} manifest-only, {len(only_annotation)} {what}-only"
    )
    if only_manifest:
        findings.problems.append(
            f"{len(only_manifest)} manifest item(s) have no entry in {what}: "
            f"{only_manifest[:_SAMPLE]}. Every selected item must be gradeable."
        )
    if only_annotation:
        findings.notes.append(
            f"  {len(only_annotation)} {what} entr(y/ies) have no manifest item "
            f"(fine under a limit): {only_annotation[:_SAMPLE]}"
        )
    return findings


def check_vocabulary(path: Path, *, what: str) -> Findings:
    findings = Findings()
    if not path.is_file():
        findings.problems.append(f"{what} not found: {path}")
        return findings
    try:
        vocab = load_vocabulary(path)
    except Exception as exc:
        findings.problems.append(f"{what} rejected: {exc}")
        return findings
    findings.notes.append(
        f"{what} OK: {path} ({vocab.name}, {len(vocab.categories)} categories, "
        f"{len(vocab.synonym_to_category)} index entries)"
    )
    return findings


def check_output_root(path: Path, *, min_free_gb: float) -> Findings:
    findings = Findings()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        findings.problems.append(f"output root not creatable: {path}: {exc}")
        return findings

    probe = path / ".preflight_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        findings.problems.append(f"output root not writable: {path}: {exc}")
        return findings

    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        findings.problems.append(
            f"only {free_gb:.1f} GB free under {path}, below the {min_free_gb:.1f} GB floor"
        )
    else:
        findings.notes.append(f"output root OK: {path} ({free_gb:.1f} GB free)")
    return findings


def check_gpu() -> Findings:
    findings = Findings()
    try:
        import torch
    except ImportError as exc:
        findings.problems.append(f"torch not importable: {exc}")
        return findings
    if not torch.cuda.is_available():
        findings.problems.append(
            "no CUDA device visible; generation would fall back to CPU and take "
            "days. Check CUDA_VISIBLE_DEVICES and the container's GPU mapping."
        )
        return findings
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    findings.notes.append(f"GPU OK: {len(names)} device(s): {', '.join(names)}")
    return findings


def check_arms(arms: list[str]) -> Findings:
    findings = Findings()
    unknown = [a for a in arms if a not in KNOWN_ARMS]
    if unknown:
        findings.problems.append(
            f"unknown arm name(s): {unknown}. Known: {list(KNOWN_ARMS)}."
        )
    else:
        findings.notes.append(f"arms OK: {arms}")
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True,
        help="Every config the run will read. The description track expands its "
             "{length} pattern into three; the composed track uses one.")
    parser.add_argument("--limit", type=int, required=True,
        help="How many manifest items the run will use.")
    parser.add_argument("--annotations", type=Path, default=None,
        help="Object annotations. Required only if the description track runs; "
             "a question-only dataset has none.")
    parser.add_argument("--vocabulary", type=Path, default=None,
        help="Object vocabulary JSON. Required only if the description track "
             "runs, same reason.")
    parser.add_argument("--questions", type=Path, default=None,
        help="Question annotations. Required only if the composed track runs.")
    parser.add_argument("--direction-terms", type=Path, default=None,
        help="Direction table. Required only if the composed track runs.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--arms", type=str, nargs="+", default=list(KNOWN_ARMS))
    parser.add_argument("--skip-gpu-check", action="store_true",
        help="Skip the CUDA check. For validating paths off the DGX only.")
    return parser


def run_checks(args) -> Findings:
    findings = Findings()

    config_findings, configs = check_configs(list(args.config))
    findings.merge(config_findings)

    manifest_ids: set[str] = set()
    if configs:
        manifest_findings, records = check_manifest(configs[0], args.limit)
        findings.merge(manifest_findings)
        if records:
            findings.merge(check_images(configs[0], records))
            manifest_ids = {r.image_id for r in records}

    # Neither track selected means nothing can be scored, whatever else is in
    # order. The two datasets each carry only one of the two ground truths, so
    # demanding both would make each of them unrunnable.
    if args.annotations is None and args.questions is None:
        findings.problems.append(
            "neither --annotations nor --questions was given: there is no "
            "ground truth of any kind, so no stage could be scored."
        )

    if args.annotations is not None:
        ann_findings, object_ids = check_object_annotations(args.annotations)
        findings.merge(ann_findings)
        if manifest_ids and object_ids:
            findings.merge(check_ids_cross(manifest_ids, object_ids, what="annotations"))
    if args.vocabulary is not None:
        findings.merge(check_vocabulary(args.vocabulary, what="vocabulary"))

    if args.questions is not None:
        q_findings, question_ids = check_question_annotations(args.questions)
        findings.merge(q_findings)
        if manifest_ids and question_ids:
            findings.merge(check_ids_cross(manifest_ids, question_ids, what="questions"))
    if args.direction_terms is not None:
        findings.merge(check_vocabulary(args.direction_terms, what="direction terms"))

    findings.merge(check_output_root(args.output_root, min_free_gb=args.min_free_gb))
    if not args.skip_gpu_check:
        findings.merge(check_gpu())
    findings.merge(check_arms(list(args.arms)))

    return findings


def main() -> int:
    args = build_parser().parse_args()

    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)

    findings = run_checks(args)

    for note in findings.notes:
        print(f"  ok    {note}")
    print()
    if findings.ok:
        print("PREFLIGHT PASSED -- nothing blocking.")
        print("=" * 78)
        return 0

    # One stream, not two: the orchestrator tees this into a log, and stderr
    # interleaved with a block-buffered stdout scrambles the report.
    print(f"PREFLIGHT FAILED -- {len(findings.problems)} problem(s):")
    for i, problem in enumerate(findings.problems, start=1):
        print(f"  {i:2d}. {problem}")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
