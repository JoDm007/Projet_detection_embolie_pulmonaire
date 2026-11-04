# backend/tests/test_services_ingestion.py — bloc services/ingestion (archives malveillantes)
import io
import zipfile

import pytest

from backend.services.ingestion import (UploadRejected, UploadTooLarge, safe_extract_dicom_zip,
                                        save_upload_stream)

LIMITS = dict(max_files=100, max_total_bytes=10 * 2**20, max_ratio=200)


def _zip(path, entries, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_extracts_only_dcm_under_generated_names(tmp_path):
    z = _zip(tmp_path / "a.zip", {"serie/IM1.dcm": b"x" * 10, "serie/IM2.DCM": b"y" * 10,
                                  "lisez-moi.txt": b"ignore"})
    n = safe_extract_dicom_zip(z, tmp_path / "out", **LIMITS)
    assert n == 2
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["00001.dcm", "00002.dcm"]


@pytest.mark.parametrize("name", ["../evil.dcm", "/abs/evil.dcm", "C:/evil.dcm", "a/../../evil.dcm"])
def test_zip_slip_rejected(tmp_path, name):
    z = _zip(tmp_path / "a.zip", {name: b"x"})
    with pytest.raises(UploadRejected):
        safe_extract_dicom_zip(z, tmp_path / "out", **LIMITS)
    assert not (tmp_path / "evil.dcm").exists()


def test_zip_bomb_ratio_rejected(tmp_path):
    z = _zip(tmp_path / "a.zip", {"bomb.dcm": b"\0" * (5 * 2**20)})
    with pytest.raises(UploadRejected, match="compression"):
        safe_extract_dicom_zip(z, tmp_path / "out", **LIMITS)


def test_total_size_limit(tmp_path):
    z = _zip(tmp_path / "a.zip", {"a.dcm": b"x" * 2048}, compression=zipfile.ZIP_STORED)
    with pytest.raises(UploadRejected, match="volumineux"):
        safe_extract_dicom_zip(z, tmp_path / "out", max_files=100, max_total_bytes=1024, max_ratio=200)


def test_too_many_files(tmp_path):
    z = _zip(tmp_path / "a.zip", {f"{i}.dcm": b"x" for i in range(5)})
    with pytest.raises(UploadRejected, match="Trop de fichiers"):
        safe_extract_dicom_zip(z, tmp_path / "out", max_files=3, max_total_bytes=2**20, max_ratio=200)


def test_no_dcm(tmp_path):
    z = _zip(tmp_path / "a.zip", {"photo.jpg": b"x"})
    with pytest.raises(UploadRejected, match="Aucun"):
        safe_extract_dicom_zip(z, tmp_path / "out", **LIMITS)


def test_upload_stream_checks_signature_and_size(tmp_path):
    with pytest.raises(UploadRejected, match="pas une archive"):
        save_upload_stream(io.BytesIO(b"GIF89a....").read, tmp_path / "u.zip", 2**20)
    buf = io.BytesIO()
    _zip(buf, {"a.dcm": b"x" * 5000}, compression=zipfile.ZIP_STORED)
    data = buf.getvalue()
    with pytest.raises(UploadTooLarge, match="volumineuse"):
        save_upload_stream(io.BytesIO(data).read, tmp_path / "u.zip", 100)
    assert save_upload_stream(io.BytesIO(data).read, tmp_path / "u.zip", 2**20) == len(data)
