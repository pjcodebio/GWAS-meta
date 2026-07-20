"""Tests for the GWAS Catalog FTP client.

Uses ``unittest.mock`` to replace ``ftplib.FTP`` so that no real network
connections are made.
"""

import ftplib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gwas_meta.gwas_client.ftp_client import GWASFTPClient


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_mock_ftp(
    nlst_return: list[str] | None = None,
    retrbinary_data: bytes = b"header\nrow1\n",
    login_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ``ftplib.FTP`` instance."""
    mock_ftp = MagicMock(spec=ftplib.FTP)
    if login_side_effect:
        mock_ftp.login.side_effect = login_side_effect
    if nlst_return is not None:
        mock_ftp.nlst.return_value = nlst_return

    def _fake_retrbinary(cmd: str, callback) -> str:
        callback(retrbinary_data)
        return "226 Transfer complete"

    mock_ftp.retrbinary.side_effect = _fake_retrbinary
    return mock_ftp


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDownloadHarmonized:
    def test_downloads_and_caches(self, tmp_path: Path) -> None:
        """File is downloaded, written to cache, and the path is returned."""
        study_id = "GCST000001"
        remote_file = (
            f"/pub/databases/gwas/summary_statistics/{study_id}"
            f"/harmonised/{study_id}.h.tsv.gz"
        )
        mock_ftp = _make_mock_ftp(
            nlst_return=[remote_file],
            retrbinary_data=b"fake\ttsv\tdata\n",
        )

        with patch("gwas_meta.gwas_client.ftp_client.ftplib.FTP", return_value=mock_ftp):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            result = client.download_harmonized(study_id)

        assert result.exists()
        assert result.name.endswith(".h.tsv.gz")
        assert result.read_bytes() == b"fake\ttsv\tdata\n"

    def test_uses_cache_on_second_call(self, tmp_path: Path) -> None:
        """A second call returns the cached file without hitting FTP."""
        study_id = "GCST000002"
        remote_file = (
            f"/pub/databases/gwas/summary_statistics/{study_id}"
            f"/harmonised/{study_id}.h.tsv.gz"
        )
        mock_ftp = _make_mock_ftp(nlst_return=[remote_file])

        with patch("gwas_meta.gwas_client.ftp_client.ftplib.FTP", return_value=mock_ftp):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            first = client.download_harmonized(study_id)
            # Reset mock call count
            mock_ftp.reset_mock()
            second = client.download_harmonized(study_id)

        assert first == second
        # FTP constructor should NOT have been called for the cached hit
        mock_ftp.retrbinary.assert_not_called()

    def test_uses_explicit_ftp_path(self, tmp_path: Path) -> None:
        """When ftp_path is given, _resolve_ftp_path is skipped."""
        study_id = "GCST000003"
        explicit = "/some/custom/path/file.h.tsv.gz"
        mock_ftp = _make_mock_ftp()

        with patch("gwas_meta.gwas_client.ftp_client.ftplib.FTP", return_value=mock_ftp):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            result = client.download_harmonized(study_id, ftp_path=explicit)

        assert result.exists()
        # nlst should not have been called because we provided ftp_path
        mock_ftp.nlst.assert_not_called()


class TestRetryOnConnectionError:
    def test_retries_then_succeeds(self, tmp_path: Path) -> None:
        """Client retries after transient FTP failures."""
        study_id = "GCST000004"
        explicit_path = "/path/to/file.h.tsv.gz"

        # First two connections fail, third succeeds
        fail_ftp = MagicMock(spec=ftplib.FTP)
        fail_ftp.retrbinary.side_effect = ftplib.error_temp("421 Temp failure")

        ok_ftp = _make_mock_ftp(retrbinary_data=b"good data\n")

        ftp_instances = [fail_ftp, fail_ftp, ok_ftp]

        with patch(
            "gwas_meta.gwas_client.ftp_client.ftplib.FTP",
            side_effect=ftp_instances,
        ), patch("gwas_meta.gwas_client.ftp_client.time.sleep"):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            result = client.download_harmonized(study_id, ftp_path=explicit_path)

        assert result.exists()
        assert result.read_bytes() == b"good data\n"

    def test_raises_after_max_retries(self, tmp_path: Path) -> None:
        """ConnectionError is raised when all retries are exhausted."""
        study_id = "GCST000005"
        explicit_path = "/path/to/file.h.tsv.gz"

        fail_ftp = MagicMock(spec=ftplib.FTP)
        fail_ftp.retrbinary.side_effect = ftplib.error_temp("421 Temp failure")

        with patch(
            "gwas_meta.gwas_client.ftp_client.ftplib.FTP",
            return_value=fail_ftp,
        ), patch("gwas_meta.gwas_client.ftp_client.time.sleep"):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            with pytest.raises(ConnectionError, match="Failed to download"):
                client.download_harmonized(study_id, ftp_path=explicit_path)


class TestResolveFtpPath:
    def test_raises_when_no_harmonized_file(self, tmp_path: Path) -> None:
        """FileNotFoundError when directory exists but has no .h.tsv.gz."""
        study_id = "GCST000006"
        mock_ftp = _make_mock_ftp(nlst_return=["readme.txt", "other.tsv.gz"])

        with patch("gwas_meta.gwas_client.ftp_client.ftplib.FTP", return_value=mock_ftp):
            client = GWASFTPClient(cache_dir=tmp_path / "cache")
            with pytest.raises(FileNotFoundError, match="No .h.tsv.gz file"):
                client.download_harmonized(study_id)
