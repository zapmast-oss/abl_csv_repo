from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
REGISTRY = CONTROL / "abl_source_registry.csv"


def main() -> int:
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_family = Counter(row["source_family"] for row in rows)
    by_authority = Counter(row["authority_level"] for row in rows)
    drives = [row for row in rows if row["can_drive_current_state"] == "yes"]
    supports = [row for row in rows if row["can_support_current_state"] == "yes"]
    sortable = [row for row in rows if row["source_family"] == "sortable_stats"]
    generated = [row for row in rows if row["source_family"] == "generated_output"]
    risky = [row for row in rows if (
        row["source_family"] in {"generated_output", "unknown"}
        or row["as_of_detection_method"].startswith("No reliable")
        or "history" in row["file_path"].lower()
    )]
    targets = sorted({row["curated_target"] for row in rows if row["curated_target"] != "none_direct"})
    payload = {
        "source_count": len(rows), "counts_by_source_family": dict(sorted(by_family.items())),
        "counts_by_authority_level": dict(sorted(by_authority.items())),
        "current_state_driver_count": len(drives), "current_state_support_count": len(supports),
        "current_state_drivers": drives, "current_state_supporters": supports,
        "sortable_stats": sortable,
        "generated_outputs_never_source_of_truth": generated,
        "stale_or_risky_sources": risky, "recommended_curated_targets": targets,
    }
    json_path = CONTROL / "abl_source_governance_report.json"
    md_path = CONTROL / "abl_source_governance_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# ABL Source Governance Report", "",
        f"Registered sources: **{len(rows)}**  ",
        f"Can drive current state: **{len(drives)}**  ",
        f"Can support current state: **{len(supports)}**", "",
        "Current-state authority is intentionally narrow: only raw OOTP game/result sources can prove the active date and results. Supplemental reports can enrich a validated capture but cannot advance its cutoff.", "",
        "## Sources by family", "", "| Family | Count |", "|---|---:|",
    ]
    lines += [f"| `{key}` | {value} |" for key, value in sorted(by_family.items())]
    lines += ["", "## Sources by authority", "", "| Authority | Count |", "|---|---:|"]
    lines += [f"| `{key}` | {value} |" for key, value in sorted(by_authority.items())]
    lines += ["", "## Files that can drive current state", ""]
    lines += [f"- `{row['file_path']}` — {row['as_of_detection_method']}" for row in drives]
    lines += ["", "## Sortable stats roles", "", "| Path | Subject | Volatility | Curated target |", "|---|---|---|---|"]
    lines += [f"| `{row['file_path']}` | {row['subject_area']} | {row['volatility']} | {row['curated_target']} |" for row in sortable]
    lines += ["", "## Generated outputs: never source of truth", "",
              f"{len(generated)} generated files are registered as `derived_output`. They may be rebuilt, audited, or rendered, but they cannot prove standings, results, or the latest game date.", "",
              "## Stale or risky sources", "",
              f"{len(risky)} sources require caution because they are derived, historical, unknown, or lack independent current-date authority. Consult the registry row before use.", "",
              "## Recommended curated current tables", ""]
    lines += [f"- `{target}`" for target in targets]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_count": len(rows), "driver_count": len(drives), "support_count": len(supports), "sortable_count": len(sortable), "markdown": str(md_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

