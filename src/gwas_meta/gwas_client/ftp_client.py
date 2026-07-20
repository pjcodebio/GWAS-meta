"""FTP client for downloading GWAS Catalog harmonized summary statistics.

The GWAS Catalog publishes harmonized summary statistics on an FTP server
at ``ftp.ebi.ac.uk``.  The typical directory layout is::

    /pub/databases/gwas/summary_statistics/{STUDY_ID}/harmonised/
        {STUDY_ID}.h.tsv.gz

This module downloads and caches those files locally.
"""

from __future__ import annotations

import ftplib
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds
_MAX_PARALLEL_DOWNLOADS = 2


class GWASFTPClient:
    """Download harmonized summary statistics from the GWAS Catalog FTP.

    Parameters
    ----------
    host : str
        FTP hostname.
    base_path : str
        Root directory on the FTP server for summary statistics.
    cache_dir : str | Path
        Local directory used to cache downloaded files.
    """

    def __init__(
        self,
        host: str = "ftp.ebi.ac.uk",
        base_path: str = "/pub/databases/gwas/summary_statistics",
        cache_dir: str | Path = ".cache/summary_stats",
    ) -> None:
        self.host = host
        self.base_path = base_path.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_harmonized(
        self,
        study_id: str,
        ftp_path: str | None = None,
        bytes_callback: callable | None = None,
    ) -> Path:
        """Download the harmonized summary-statistics file for a study.

        If the file has already been cached locally the download is skipped.

        Parameters
        ----------
        study_id : str
            GWAS Catalog study accession (e.g. ``"GCST000001"``).
        ftp_path : str | None
            Explicit path to the ``.h.tsv.gz`` file on the FTP server.
            When *None* the path is resolved automatically.
        bytes_callback : callable | None
            Optional callable receiving ``(n_bytes)`` after each chunk is
            written to disk.  Used for progress tracking.

        Returns
        -------
        Path
            Local filesystem path to the downloaded (cached) file.

        Raises
        ------
        FileNotFoundError
            If the harmonized file cannot be located on the server.
        ConnectionError
            If the FTP server cannot be reached after retries.
        """
        cached = self._cached_path(study_id)
        if cached is not None:
            logger.info("Using cached file for %s: %s", study_id, cached)
            return cached

        # Resolve and download using a single FTP connection
        ftp = self._connect()
        try:
            remote_path = ftp_path or self._resolve_ftp_path(study_id, ftp)
            filename = Path(remote_path).name
            local_path = self.cache_dir / study_id / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self._download_with_retries(
                remote_path, local_path, ftp, bytes_callback=bytes_callback
            )
        finally:
            try:
                ftp.quit()
            except Exception:
                pass

        logger.info("Downloaded %s -> %s", remote_path, local_path)
        return local_path

    def _get_file_size(self, ftp: ftplib.FTP, remote_path: str) -> int | None:
        """Query the size of a remote file, or return None on failure."""
        try:
            ftp.voidcmd("TYPE I")  # binary mode required for SIZE
            return ftp.size(remote_path)
        except (ftplib.error_perm, ftplib.error_temp, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _study_range_bucket(study_id: str) -> str:
        """Compute the range-bucket directory for a study accession.

        The GWAS Catalog FTP organises studies into directories like
        ``GCST004001-GCST005000``.  The numeric suffix determines the
        bucket.
        """
        # e.g. "GCST004988" -> prefix="GCST", num=4988
        # e.g. "GCST90011804" -> prefix="GCST", num=90011804
        import re
        m = re.match(r"^([A-Z]+)(\d+)$", study_id)
        if not m:
            return study_id  # fallback: no bucketing
        prefix, num_str = m.group(1), m.group(2)
        num = int(num_str)
        bucket_size = 1000
        lower = ((num - 1) // bucket_size) * bucket_size + 1
        upper = lower + bucket_size - 1
        # Preserve leading-zero width from original accession
        width = len(num_str)
        return f"{prefix}{lower:0{width}d}-{prefix}{upper:0{width}d}"

    def _resolve_ftp_path(
        self, study_id: str, ftp: ftplib.FTP | None = None
    ) -> str:
        """List the harmonised directory and find the ``.h.tsv.gz`` file."""
        bucket = self._study_range_bucket(study_id)
        harmonised_dir = f"{self.base_path}/{bucket}/{study_id}/harmonised"
        own_connection = ftp is None
        if own_connection:
            ftp = self._connect()
        try:
            files = ftp.nlst(harmonised_dir)
        except ftplib.error_perm as exc:
            raise FileNotFoundError(
                f"Could not list {harmonised_dir}: {exc}"
            ) from exc
        finally:
            if own_connection:
                ftp.quit()

        for f in files:
            if f.endswith(".h.tsv.gz"):
                return f

        raise FileNotFoundError(
            f"No .h.tsv.gz file found in {harmonised_dir} "
            f"(files: {files})"
        )

    def _download_with_retries(
        self,
        remote_path: str,
        local_path: Path,
        ftp: ftplib.FTP | None = None,
        bytes_callback: callable | None = None,
    ) -> None:
        """Download *remote_path* to *local_path* with retry logic.

        Streams directly to disk to avoid holding the entire file in memory.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                own_connection = ftp is None
                if own_connection:
                    ftp = self._connect()
                try:
                    with open(local_path, "wb") as f:
                        def _write_chunk(data: bytes) -> None:
                            f.write(data)
                            if bytes_callback:
                                bytes_callback(len(data))
                        ftp.retrbinary(f"RETR {remote_path}", _write_chunk)
                    return
                finally:
                    if own_connection:
                        ftp.quit()
            except (*ftplib.all_errors, OSError) as exc:
                last_exc = exc
                # Remove partial file
                local_path.unlink(missing_ok=True)
                logger.warning(
                    "FTP download attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                # Connection may be broken, get a fresh one for retry
                ftp = None
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

        raise ConnectionError(
            f"Failed to download {remote_path} after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _connect(self) -> ftplib.FTP:
        """Open an anonymous FTP connection to the host."""
        ftp = ftplib.FTP(self.host, timeout=60)
        ftp.login()
        return ftp

    def _cached_path(self, study_id: str) -> Path | None:
        """Return the path to a previously cached file, or *None*."""
        study_dir = self.cache_dir / study_id
        if not study_dir.is_dir():
            return None
        for child in study_dir.iterdir():
            if child.name.endswith(".h.tsv.gz"):
                return child
        return None
