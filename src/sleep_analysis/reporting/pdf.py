"""PDF reporting utilities for weekly snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Compatibility shim for OpenSSL builds that do not accept usedforsecurity kwarg
try:  # pragma: no cover - environment-specific behaviour
    from hashlib import md5 as _md5
    from reportlab.pdfbase import pdfdoc

    def _compat_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _md5(*args, **kwargs)

    pdfdoc.md5 = _compat_md5  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


def _collect_metrics(reports_dir: Path) -> List[Dict[str, str]]:
    metrics: List[Dict[str, str]] = []
    for path in sorted(reports_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics.append(
            {
                "file": path.name,
                "model": payload.get("model_name") or payload.get("model"),
                "target": payload.get("target"),
                "mae": f"{payload.get('metrics', {}).get('mae', 'n/a')}",
                "mape": f"{payload.get('metrics', {}).get('mape', 'n/a')}",
                "coverage": f"{payload.get('metrics', {}).get('coverage', 'n/a')}",
            }
        )
    return metrics


def generate_weekly_pdf(
    reports_dir: Path,
    output_dir: Path,
    *,
    week: Optional[str] = None,
) -> Path:
    """Generate a weekly PDF summary from model reports."""

    reports_dir = reports_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _collect_metrics(reports_dir)

    iso_week = week or datetime.utcnow().strftime("%Y-W%V")
    pdf_path = output_dir / f"weekly-report-{iso_week}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER
    margin = 1 * inch
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Sleep Pattern Storyboard — Weekly Report")
    y -= 0.4 * inch
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Week: {iso_week}")
    y -= 0.3 * inch
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 0.5 * inch

    if not metrics:
        c.drawString(margin, y, "No model reports available.")
    else:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Model Metrics:")
        y -= 0.3 * inch
        c.setFont("Helvetica", 11)
        for entry in metrics:
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 11)
            line = (
                f"- {entry['model']} ({entry['target']}): "
                f"MAE={entry['mae']} MAPE={entry['mape']} Coverage={entry['coverage']}"
            )
            c.drawString(margin, y, line)
            y -= 0.25 * inch

    c.showPage()
    c.save()
    return pdf_path
