"""Convert Group-X-Report.md -> .docx and .pdf (PDF via fpdf2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Group-X-Report.md"
DOCX_OUT = ROOT / "Group-X-Report.docx"
PDF_OUT = ROOT / "Group-X-Report.pdf"


def build_docx(text: str) -> None:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    def set_run_font(run, *, ascii_font="Calibri", size=None, bold=None):
        run.font.name = ascii_font
        r_pr = run._element.get_or_add_rPr()
        r_fonts = r_pr.get_or_add_rFonts()
        r_fonts.set(qn("w:eastAsia"), ascii_font)
        if size is not None:
            run.font.size = Pt(size)
        if bold is not None:
            run.bold = bold

    def add_inline(paragraph, raw: str, size: int = 11) -> None:
        pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
        for part in pattern.split(raw):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                run = paragraph.add_run(part[2:-2])
                set_run_font(run, size=size, bold=True)
            elif part.startswith("`") and part.endswith("`"):
                run = paragraph.add_run(part[1:-1])
                set_run_font(run, ascii_font="Consolas", size=size)
            else:
                run = paragraph.add_run(part)
                set_run_font(run, size=size)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("```"):
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            run = p.add_run("\n".join(buf))
            set_run_font(run, ascii_font="Consolas", size=9)
        elif line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\|?\s*:?-{3,}", lines[i + 1].strip()
        ):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if re.search(r"-{3,}", lines[i]):
                    i += 1
                    continue
                rows.append(
                    [c.strip() for c in lines[i].strip().strip("|").split("|")]
                )
                i += 1
            if rows:
                table = doc.add_table(rows=len(rows), cols=len(rows[0]))
                table.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, cell in enumerate(row):
                        table.rows[r].cells[c].text = cell
            continue
        elif line.strip() in {"", "---"}:
            pass
        elif line.startswith("> "):
            p = doc.add_paragraph()
            add_inline(p, line[2:].strip(), size=10)
        else:
            p = doc.add_paragraph()
            add_inline(p, line)
        i += 1

    doc.save(DOCX_OUT)
    print(f"Wrote {DOCX_OUT}")


def build_pdf(text: str) -> None:
    from fpdf import FPDF

    class PDF(FPDF):
        def footer(self) -> None:
            self.set_y(-15)
            self.set_font("Helvetica", size=9)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    def clean(s: str) -> str:
        # fpdf core fonts are Latin-1; strip unsupported chars lightly
        return (
            s.replace("→", "->")
            .replace("—", "-")
            .replace("–", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("×", "x")
            .replace("≤", "<=")
            .replace("≥", ">=")
            .replace("✓", "[OK]")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )

    in_code = False
    skip_mermaid = False
    usable = pdf.epw  # effective page width

    def write_wrapped(font: str, style: str, size: int, content: str, lh: float) -> None:
        pdf.set_font(font, style, size)
        # Hard-wrap very long tokens so fpdf never gets zero-width cells.
        safe = content if content.strip() else " "
        while pdf.get_string_width(safe) > usable and len(safe) > 1:
            # insert breaks for long unbroken strings
            cut = max(1, int(len(safe) * usable / max(pdf.get_string_width(safe), 1)))
            pdf.multi_cell(usable, lh, safe[:cut])
            safe = safe[cut:]
        pdf.multi_cell(usable, lh, safe)

    for line in text.splitlines():
        if line.startswith("```"):
            fence = line.strip().strip("`").strip().lower()
            if not in_code and fence.startswith("mermaid"):
                skip_mermaid = True
                in_code = True
                write_wrapped(
                    "Helvetica",
                    "I",
                    10,
                    "[Flowchart: see Group-X-Report.md mermaid diagram / Overleaf figure]",
                    5,
                )
                continue
            if in_code and skip_mermaid:
                skip_mermaid = False
                in_code = False
                continue
            in_code = not in_code
            continue
        if skip_mermaid:
            continue
        raw = clean(line)
        if not raw.strip():
            pdf.ln(4)
            continue
        if in_code:
            write_wrapped("Courier", "", 8, raw[:200], 4)
            continue
        if line.startswith("# "):
            write_wrapped("Helvetica", "B", 16, clean(line[2:]), 8)
            pdf.ln(2)
        elif line.startswith("## "):
            write_wrapped("Helvetica", "B", 13, clean(line[3:]), 7)
            pdf.ln(1)
        elif line.startswith("### "):
            write_wrapped("Helvetica", "B", 12, clean(line[4:]), 6)
        elif line.startswith("> "):
            write_wrapped("Helvetica", "I", 10, clean(line[2:]), 5)
        elif line.strip().startswith("|"):
            write_wrapped("Courier", "", 7, raw[:180], 4)
        else:
            plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
            plain = re.sub(r"`([^`]+)`", r"\1", plain)
            write_wrapped("Helvetica", "", 11, plain, 5)

    pdf.output(str(PDF_OUT))
    print(f"Wrote {PDF_OUT}")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    build_docx(text)
    try:
        build_pdf(text)
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "fpdf2", "-q"]
        )
        build_pdf(text)


if __name__ == "__main__":
    main()
