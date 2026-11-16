"""Export reports/report.docx -> reports/report.pdf and reports/slides.pptx -> reports/slides.pdf via Office COM."""

from __future__ import annotations

from pathlib import Path
import win32com.client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def export_docx_to_pdf() -> None:
    docx_path = (REPORTS_DIR / "report.docx").resolve()
    pdf_path = (REPORTS_DIR / "report.pdf").resolve()
    if not docx_path.exists():
        return
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx_path))
        try:
            doc.TablesOfContents(1).Update()
        except Exception:
            pass
        doc.Save()
        # wdFormatPDF = 17
        doc.SaveAs(str(pdf_path), FileFormat=17)
        doc.Close(SaveChanges=False)
        print(f"Exported {pdf_path}")
    finally:
        word.Quit()


def export_pptx_to_pdf() -> None:
    pptx_path = (REPORTS_DIR / "slides.pptx").resolve()
    pdf_path = (REPORTS_DIR / "slides.pdf").resolve()
    if not pptx_path.exists():
        return
    try:
        ppt = win32com.client.DispatchEx("PowerPoint.Application")
        try:
            pres = ppt.Presentations.Open(str(pptx_path), WithWindow=False)
            # ppSaveAsPDF = 32
            pres.SaveAs(str(pdf_path), 32)
            pres.Close()
            print(f"Exported {pdf_path}")
        finally:
            ppt.Quit()
    except Exception as e:
        print(f"PowerPoint PDF export skipped/note: {e}")


if __name__ == "__main__":
    export_docx_to_pdf()
    export_pptx_to_pdf()
