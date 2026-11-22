"""Convert REPORT_ZH.md to a Word document."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT_ZH.md"
OUT = ROOT / "REPORT_ZH.docx"


def set_run_font(run, *, east_asia: str = "Microsoft JhengHei", ascii_font: str = "Calibri", size: int | None = None, bold: bool | None = None):
    run.font.name = ascii_font
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def add_inline_markdown(paragraph, text: str, *, size: int = 11, bold_default: bool = False):
    """Render **bold**, `code`, and plain text into a paragraph."""
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    parts = pattern.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=size, bold=True)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, ascii_font="Consolas", east_asia="Consolas", size=size, bold=bold_default)
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size, bold=bold_default)


def add_paragraph(doc: Document, text: str, *, style: str | None = None, size: int = 11):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    add_inline_markdown(p, text, size=size)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.25
    return p


def parse_table_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def is_table_sep(line: str) -> bool:
    s = line.strip().strip("|").replace(":", "").replace("-", "").replace("|", "").replace(" ", "")
    return line.strip().startswith("|") and s == ""


def convert() -> Path:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")

    i = 0
    in_code = False
    code_lang = ""
    code_buf: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_buf = []
            else:
                in_code = False
                block = "\n".join(code_buf)
                if code_lang == "mermaid":
                    p = doc.add_paragraph()
                    run = p.add_run("【流程圖說明】")
                    set_run_font(run, size=11, bold=True)
                    add_paragraph(
                        doc,
                        "使用者問題 → 自動選技能 → ReAct 迴圈（最多 10 步）→ 呼叫 DeepSeek → "
                        "（若要工具）search / visit / find_in_doc → 壓縮結果寫入狀態 → 再迴圈；"
                        "（若純文字答案）檢查是否像最終答案 → 是則輸出，否則繼續迴圈。",
                    )
                    note = doc.add_paragraph()
                    run = note.add_run("（原始 mermaid 流程定義如下，方便對照）")
                    set_run_font(run, size=9, bold=False)
                p = doc.add_paragraph()
                run = p.add_run(block if block else "(empty)")
                set_run_font(run, ascii_font="Consolas", east_asia="Consolas", size=9)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.left_indent = Cm(0.3)
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        if line.strip() == "---":
            p = doc.add_paragraph()
            run = p.add_run("─" * 28)
            set_run_font(run, size=9)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        if line.startswith("# "):
            p = doc.add_heading(line[2:].strip(), level=0)
            for run in p.runs:
                set_run_font(run, size=18, bold=True)
            i += 1
            continue

        if line.startswith("## "):
            p = doc.add_heading(line[3:].strip(), level=1)
            for run in p.runs:
                set_run_font(run, size=14, bold=True)
            i += 1
            continue

        if line.startswith("### "):
            p = doc.add_heading(line[4:].strip(), level=2)
            for run in p.runs:
                set_run_font(run, size=12, bold=True)
            i += 1
            continue

        # Markdown table
        if line.strip().startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            headers = parse_table_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Table Grid"
            for c, h in enumerate(headers):
                cell = table.rows[0].cells[c]
                cell.text = ""
                p = cell.paragraphs[0]
                add_inline_markdown(p, h, size=10, bold_default=True)
            for r, row in enumerate(rows):
                for c in range(len(headers)):
                    cell = table.rows[r + 1].cells[c]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    val = row[c] if c < len(row) else ""
                    add_inline_markdown(p, val, size=10)
            doc.add_paragraph()
            continue

        # Lists
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_inline_markdown(p, m.group(2), size=11)
            i += 1
            continue

        if line.lstrip().startswith("- "):
            content = line.lstrip()[2:]
            p = doc.add_paragraph(style="List Bullet")
            add_inline_markdown(p, content, size=11)
            i += 1
            continue

        add_paragraph(doc, line.strip(), size=11)
        i += 1

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = convert()
    print(out)
