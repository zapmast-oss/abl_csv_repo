"""Central path policy for ABL generated outputs and protected inputs."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_INPUT_RELATIVE_ROOTS = (
    PurePosixPath("csv/ootp_csv"),
    PurePosixPath("csv/abl_statistics"),
    PurePosixPath("csv/statsplus/current"),
    PurePosixPath("csv/in/almanac_core"),
)
ACCIDENTAL_OUTPUT_RELATIVE_ROOTS = (
    PurePosixPath("out"),
    PurePosixPath("csv/csv"),
)


class PathPolicyError(ValueError):
    """Raised when generated output targets a protected or accidental path."""


def _resolved(path: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    root = Path(repo_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def protected_input_roots(repo_root: str | Path = REPO_ROOT) -> tuple[Path, ...]:
    root = Path(repo_root).resolve()
    return tuple(_resolved(path, root) for path in PROTECTED_INPUT_RELATIVE_ROOTS)


def accidental_output_roots(repo_root: str | Path = REPO_ROOT) -> tuple[Path, ...]:
    root = Path(repo_root).resolve()
    return tuple(_resolved(path, root) for path in ACCIDENTAL_OUTPUT_RELATIVE_ROOTS)


def validate_output_path(path: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    """Return a resolved output path or reject protected/sprawl destinations.

    This check is intentionally valid for paths that do not exist yet, so callers
    can validate before creating a parent directory or opening a file for write.
    """

    root = Path(repo_root).resolve()
    candidate = _resolved(path, root)
    for protected in protected_input_roots(root):
        if _is_within(candidate, protected):
            raise PathPolicyError(
                f"Generated output is forbidden beneath protected input root "
                f"{protected}: {candidate}"
            )
    for accidental in accidental_output_roots(root):
        if _is_within(candidate, accidental):
            raise PathPolicyError(
                f"Generated output is forbidden beneath accidental output root "
                f"{accidental}: {candidate}"
            )
    return candidate


def validate_output_paths(
    paths: Iterable[str | Path], repo_root: str | Path = REPO_ROOT
) -> tuple[Path, ...]:
    return tuple(validate_output_path(path, repo_root) for path in paths)


def _repo_path(path: str | Path) -> PurePosixPath:
    return PurePosixPath(str(path).replace("\\", "/").lstrip("./"))


def is_authoritative_ootp_csv(path: str | Path) -> bool:
    """True only for an immediate CSV child of csv/ootp_csv."""

    rel = _repo_path(path)
    return (
        rel.parent == PurePosixPath("csv/ootp_csv")
        and rel.suffix.lower() == ".csv"
    )


def is_immediate_sortable_csv(path: str | Path) -> bool:
    """True only for an immediate CSV child of csv/abl_statistics."""

    rel = _repo_path(path)
    return (
        rel.parent == PurePosixPath("csv/abl_statistics")
        and rel.suffix.lower() == ".csv"
    )


def existing_accidental_output_roots(
    repo_root: str | Path = REPO_ROOT,
) -> tuple[Path, ...]:
    """Report known legacy sprawl without modifying it."""

    root = Path(repo_root).resolve()
    candidates = (*accidental_output_roots(root), root / "csv" / "ootp_csv" / "out")
    return tuple(path for path in candidates if path.exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a proposed ABL generated-output path without writing it."
    )
    parser.add_argument("output_path", help="Proposed file or directory output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(validate_output_path(args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
