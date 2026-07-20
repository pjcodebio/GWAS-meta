"""GWAS Catalog client package -- REST and FTP access."""

from gwas_meta.gwas_client.ftp_client import GWASFTPClient
from gwas_meta.gwas_client.models import GWASStudy, GWASTrait
from gwas_meta.gwas_client.rest_client import GWASCatalogClient

__all__ = [
    "GWASCatalogClient",
    "GWASFTPClient",
    "GWASStudy",
    "GWASTrait",
]
