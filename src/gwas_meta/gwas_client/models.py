"""Data models for the GWAS Catalog client."""

from dataclasses import dataclass, field


@dataclass
class GWASTrait:
    """An EFO trait from the GWAS Catalog.

    Parameters
    ----------
    efo_id : str
        EFO identifier (e.g. "EFO_0000270").
    trait_name : str
        Human-readable trait name (e.g. "asthma").
    uri : str
        Full URI for the EFO term.
    """

    efo_id: str
    trait_name: str
    uri: str


@dataclass
class GWASStudy:
    """A study record from the GWAS Catalog.

    Parameters
    ----------
    study_id : str
        GWAS Catalog accession (e.g. "GCST000001").
    title : str
        Publication title.
    publication : str
        First author and publication year string.
    pub_date : str
        Publication date in ISO format.
    journal : str
        Journal name.
    initial_sample_size : str
        Description of the initial sample (e.g. "1,000 European ancestry cases").
    traits : list[GWASTrait]
        EFO traits associated with this study.
    has_summary_stats : bool
        Whether full summary statistics are available for download.
    ftp_path : str | None
        FTP path to the summary statistics, if available.
    """

    study_id: str
    title: str
    publication: str
    pub_date: str
    journal: str
    initial_sample_size: str
    traits: list[GWASTrait] = field(default_factory=list)
    has_summary_stats: bool = False
    ftp_path: str | None = None
    pubmed_id: str | None = None
