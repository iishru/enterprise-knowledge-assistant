# test_loader.py - checks that files load with the right source/page metadata.
from pathlib import Path

import pytest

from src.loader import clean_text, load_directory, load_file

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc\n\n\n\nd  ") == "a b c\n\nd"


def test_load_markdown_has_source_and_no_page(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nSome text.", encoding="utf-8")
    docs = load_file(f)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "note.md"}


def test_load_pdf_has_one_based_page_numbers():
    docs = load_file(SAMPLE_DIR / "leave_policy.pdf")
    assert len(docs) >= 1
    assert docs[0].metadata["source"] == "leave_policy.pdf"
    assert docs[0].metadata["page"] == 1
    assert "Casual Leave" in " ".join(d.page_content for d in docs)


def test_load_docx():
    docs = load_file(SAMPLE_DIR / "employee_onboarding_guide.docx")
    assert docs[0].metadata["source"] == "employee_onboarding_guide.docx"
    assert "onboarding buddy" in docs[0].page_content


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "pic.png"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(f)


def test_load_directory_skips_unsupported_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"x")
    docs = load_directory(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["a.txt"]


def test_load_directory_sample_docs_covers_all_eight_files():
    docs = load_directory(SAMPLE_DIR)
    assert len({d.metadata["source"] for d in docs}) == 8
    assert all("/" not in d.metadata["source"] for d in docs)  # file name only
