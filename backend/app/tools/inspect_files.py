"""Deterministic inspection of uploaded files, so the planner sees real structure (columns, placeholders, labels) instead of guessing."""
from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}|<<\s*([^<>]+?)\s*>>|\[\[\s*([^\[\]]+?)\s*\]\]|\$\{\s*([^{}]+?)\s*\}|\{\s*([A-Za-z_][\w \-']{0,40}?)\s*\}")
BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,60})\]")
LABELED_BRACKET_RE = re.compile(r"([A-Za-z][A-Za-z0-9 .'/&()\-]{1,45}?)\s*:\s*(\[[^\[\]\n]{0,60}\])")
LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .'/&()\-]{1,45}?)\s*:\s*(?:_{2,}|\.{3,}|\s*)$")


def norm_key(s: str) -> str:
    """Case-, space- and punctuation-insensitive key: "Father's Name:" == "FATHERS NAME" == "[Father's Name]"."""
    return re.sub(r"[\W_]+", "", s.lower().strip().strip("[]"))


def find_placeholders(text: str) -> list[str]:
    out = []
    for m in PLACEHOLDER_RE.finditer(text):
        k = next(g for g in m.groups() if g)
        if k not in out:
            out.append(k.strip())
    return out


def find_bracket_fields(text: str) -> dict:
    """Template fields written as bracketed text: 'Student Name: [STUDENT NAME]' -> labelled; '[SCHOOL NAME]' -> bare token."""
    labelled, bare, seen = [], [], set()
    for m in LABELED_BRACKET_RE.finditer(text):
        lab = m.group(1).strip()
        if lab.lower() not in seen:
            seen.add(lab.lower())
            labelled.append({"label": lab, "token": m.group(2)})
    for m in BRACKET_RE.finditer(text):
        tok = m.group(1).strip()
        if re.search(r"[A-Za-z]{3,}", tok) and tok not in bare:
            bare.append(tok)
    return {"bracket_fields": labelled[:40], "bracket_tokens": bare[:40]}


def find_grid_fields(tables) -> list[str]:
    """'RowLabel / ColumnHeader' for every blank or bracketed body cell of a table with a header row (usable as template.fill keys)."""
    out: list[str] = []
    for t in tables:
        rows = t.rows
        if len(rows) < 2:
            continue
        header = [c.text.strip() for c in rows[0].cells]
        if sum(1 for h in header if h) < 2:
            continue
        for row in rows[1:]:
            cells = row.cells
            label = cells[0].text.strip()
            for j in range(1, len(cells)):
                txt = cells[j].text.strip()
                if label and header[j] and not re.search(r"[A-Za-z]", txt):
                    out.append(f"{label} / {header[j]}")
    return list(dict.fromkeys(out))[:80]


def _clean(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def read_table(path: Path, sheet: str | None = None, header_row: int | None = None, limit: int = 5000) -> dict:
    """Read an xlsx/csv into {columns, rows}. The header row is auto-detected (first row with >=2 filled cells) unless given."""
    import pandas as pd

    ext = path.suffix.lower()
    if ext == ".csv":
        raw = pd.read_csv(path, header=None, dtype=object, keep_default_na=False)
        sheet_name = None
        sheets = None
    elif ext == ".xlsx":
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
        sheet_name = sheet if sheet in sheets else sheets[0]
        if sheet and sheet not in sheets:
            raise ValueError(f"sheet '{sheet}' not found; available sheets: {sheets}")
        raw = xl.parse(sheet_name, header=None, dtype=object)
    else:
        raise ValueError(f"unsupported table type {ext}")
    raw = raw.dropna(how="all")
    if raw.empty:
        return {"columns": [], "rows": [], "row_count": 0, "sheet": sheet_name, "sheets": sheets, "header_row": None}
    if header_row is None:
        header_row = next((i for i, (_, r) in enumerate(raw.iterrows()) if sum(1 for v in r if str(v).strip() not in ("", "nan", "None")) >= 2), 0)
    hdr = raw.iloc[header_row]
    cols, seen = [], {}
    for i, v in enumerate(hdr):
        name = str(v).strip() if str(v).strip() not in ("", "nan", "None") else f"Column{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        cols.append(name if seen[name] == 1 else f"{name}.{seen[name] - 1}")
    body = raw.iloc[header_row + 1:]
    rows = []
    for _, r in body.iterrows():
        d = {c: _clean(v) for c, v in zip(cols, r)}
        if any(v not in (None, "") for v in d.values()):
            rows.append(d)
    return {"columns": cols, "rows": rows[:limit], "row_count": len(rows), "truncated": len(rows) > limit, "sheet": sheet_name,
            "sheets": sheets, "header_row": int(header_row)}


def inspect(path: Path, display_name: str) -> dict:
    ext = path.suffix.lower()
    info: dict = {"name": display_name, "type": ext.lstrip(".")}
    try:
        if ext in (".xlsx", ".csv"):
            t = read_table(path, limit=5)
            info.update(kind="table", columns=t["columns"], row_count=t["row_count"], sample_rows=t["rows"][:3], sheets=t["sheets"], sheet=t["sheet"])
            if t["sheets"] and len(t["sheets"]) > 1:  # show the planner every sheet, with its own columns
                details = []
                for name in t["sheets"][:8]:
                    s = read_table(path, sheet=name, limit=2)
                    details.append({"sheet": name, "columns": s["columns"], "row_count": s["row_count"], "sample_rows": s["rows"][:2]})
                info["sheet_details"] = details
        elif ext == ".docx":
            import docx

            d = docx.Document(str(path))
            texts = [p.text for p in d.paragraphs if p.text.strip()]
            cells = [c.text for t in d.tables for r in t.rows for c in r.cells]
            for sec in d.sections:
                texts += [p.text for p in sec.header.paragraphs + sec.footer.paragraphs if p.text.strip()]
            alltext = "\n".join(texts + cells)
            labels = [m.group(1).strip() for s in texts + cells for m in [LABEL_RE.match(s.strip())] if m]
            info.update(kind="word_template", paragraphs=len(texts), tables=len(d.tables), placeholders=find_placeholders(alltext),
                        labels=list(dict.fromkeys(labels))[:40], preview=alltext[:600], grid_fields=find_grid_fields(d.tables), **find_bracket_fields(alltext))
        elif ext == ".pdf":
            import pymupdf

            with pymupdf.open(path) as doc:
                text = "\n".join(p.get_text() for p in doc)
                widgets = [w.field_name for p in doc for w in (p.widgets() or []) if w.field_name]
            labels = [m.group(1).strip() for line in text.splitlines() for m in [LABEL_RE.match(line.strip())] if m]
            info.update(kind="pdf", placeholders=find_placeholders(text), form_fields=widgets,
                        labels=list(dict.fromkeys(labels))[:40], preview=text[:600], has_text=bool(text.strip()), **find_bracket_fields(text))
            with pymupdf.open(path) as doc:
                info["pages"] = doc.page_count
        elif ext in (".png", ".jpg", ".jpeg"):
            from PIL import Image

            with Image.open(path) as im:
                info.update(kind="image", size=list(im.size), note="images cannot be filled automatically (no OCR/layout detection)")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            info.update(kind="text", chars=len(text), preview=text[:600])
        if ext == ".xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(path, data_only=False)
            ph, labels = [], []
            for ws in wb.worksheets:
                for row in ws.iter_rows():
                    for c in row:
                        if isinstance(c.value, str):
                            ph += find_placeholders(c.value)
                            m = LABEL_RE.match(c.value.strip())
                            if m:
                                labels.append(m.group(1).strip())
            info.update(placeholders=list(dict.fromkeys(ph)), labels=list(dict.fromkeys(labels))[:40])
    except Exception as e:  # noqa: BLE001 - an unreadable file is reported, not fatal
        info.update(kind="unreadable", error=f"{type(e).__name__}: {e}")
    return info
