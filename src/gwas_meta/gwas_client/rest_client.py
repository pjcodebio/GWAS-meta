"""REST client for the GWAS Catalog API.

The GWAS Catalog exposes a HAL+JSON API at
https://www.ebi.ac.uk/gwas/rest/api.  Studies, traits and associations
are returned inside ``_embedded`` and paginated via ``_links.next``.
"""

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gwas_meta.gwas_client.models import GWASStudy, GWASTrait

logger = logging.getLogger(__name__)


class GWASCatalogClient:
    """Thin wrapper around the GWAS Catalog REST API.

    Parameters
    ----------
    base_url : str
        Root URL of the API (no trailing slash).
    timeout : int
        HTTP request timeout in seconds.
    rate_limit_delay : float
        Minimum delay in seconds between consecutive requests.
    """

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/gwas/rest/api",
        timeout: int = 90,
        rate_limit_delay: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def search_traits(self, query: str) -> list[GWASTrait]:
        """Search EFO traits by name.

        Uses ``/efoTraits/search/findByEfoTrait?trait={query}``.
        """
        url = f"{self.base_url}/efoTraits/search/findByEfoTrait"
        params = {"trait": query}
        raw_items = self._get_paginated(url, params)
        traits: list[GWASTrait] = []
        for item in raw_items:
            trait = self._parse_trait(item)
            if trait is not None:
                traits.append(trait)
        logger.info("search_traits(%r) returned %d traits", query, len(traits))
        return traits

    def search_studies(self, query: str, size: int = 50) -> list[GWASStudy]:
        """Search studies by disease-trait name.

        Uses ``/studies/search/findByDiseaseTrait?diseaseTraitContains={query}``.
        """
        url = f"{self.base_url}/studies/search/findByDiseaseTrait"
        params = {"diseaseTrait": query, "size": size}
        raw_items = self._get_paginated(url, params)
        studies: list[GWASStudy] = []
        for item in raw_items:
            study = self._parse_study(item)
            if study is not None:
                studies.append(study)
        logger.info("search_studies(%r) returned %d studies", query, len(studies))
        return studies

    def search_by_efo(self, efo_id: str, size: int = 100) -> list[GWASStudy]:
        """Retrieve studies associated with an EFO trait.

        Parameters
        ----------
        efo_id : str
            Either a short EFO id (``"EFO_0000270"``) or a full URI.
        size : int
            Page size for the request.
        """
        if efo_id.startswith("http"):
            efo_uri = efo_id
        else:
            efo_uri = f"http://www.ebi.ac.uk/efo/{efo_id}"

        url = f"{self.base_url}/studies/search/findByEfoUri"
        params = {"uri": efo_uri, "size": size}
        raw_items = self._get_paginated(url, params)
        studies: list[GWASStudy] = []
        for item in raw_items:
            study = self._parse_study(item)
            if study is not None:
                studies.append(study)
        logger.info("search_by_efo(%r) returned %d studies", efo_id, len(studies))
        return studies

    def get_study(self, study_id: str) -> GWASStudy:
        """Fetch a single study by its GWAS Catalog accession.

        Raises
        ------
        requests.HTTPError
            If the study does not exist or the server returns an error.
        """
        url = f"{self.base_url}/studies/{study_id}"
        data = self._get_json(url)
        study = self._parse_study(data)
        if study is None:
            raise ValueError(f"Could not parse study {study_id}")
        return study

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def _get_paginated(
        self,
        url: str,
        params: dict | None = None,
        max_pages: int = 10,
    ) -> list[dict]:
        """Follow HAL ``_links.next`` until exhausted or *max_pages* reached.

        Returns the concatenated list of items found under
        ``_embedded.<resource>`` across all pages.
        """
        all_items: list[dict] = []
        current_url: str | None = url
        current_params = params
        page = 0

        while current_url is not None and page < max_pages:
            data = self._get_json(current_url, params=current_params)
            # After the first request params are baked into _links.next
            current_params = None

            embedded = data.get("_embedded", {})
            # The resource key varies (studies, efoTraits, …); grab them all.
            for key, items in embedded.items():
                if isinstance(items, list):
                    all_items.extend(items)

            # Follow next link if present
            next_href = (
                data.get("_links", {}).get("next", {}).get("href")
            )
            current_url = next_href
            page += 1

        return all_items

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        """Issue a GET request and return the parsed JSON body."""
        time.sleep(self.rate_limit_delay)
        logger.debug("GET %s params=%s", url, params)
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_trait(data: dict) -> GWASTrait | None:
        """Parse a single EFO-trait JSON object into a :class:`GWASTrait`."""
        try:
            short_form = data.get("shortForm", "")
            trait_name = data.get("trait", "")
            uri = data.get("uri", "")
            return GWASTrait(efo_id=short_form, trait_name=trait_name, uri=uri)
        except (KeyError, TypeError):
            logger.warning("Failed to parse trait: %s", data)
            return None

    @staticmethod
    def _parse_study(data: dict) -> GWASStudy | None:
        """Parse a single study JSON object into a :class:`GWASStudy`."""
        try:
            study_id = data.get("accessionId", "")
            publication_info = data.get("publicationInfo", {})
            title = publication_info.get("title", data.get("title", ""))
            pub_date = publication_info.get("publicationDate", data.get("publicationDate", ""))
            author = publication_info.get("author", {})
            if isinstance(author, dict):
                publication = author.get("fullname", "")
            else:
                publication = str(author)
            journal = publication_info.get("publication", data.get("publication", ""))

            initial_sample_size = data.get("initialSampleSize", "")
            has_summary_stats = data.get("fullPvalueSet", False)

            # Parse embedded EFO traits if present
            traits: list[GWASTrait] = []
            efo_traits = data.get("_embedded", {}).get("efoTraits", [])
            for t in efo_traits:
                parsed = GWASCatalogClient._parse_trait(t)
                if parsed is not None:
                    traits.append(parsed)

            # Attempt to extract FTP path from _links or platform info
            ftp_path: str | None = None
            platform_info = data.get("platformInfo", data.get("genotypingTechnologies", []))
            if isinstance(platform_info, str) and platform_info.startswith("ftp"):
                ftp_path = platform_info

            pubmed_id = publication_info.get("pubmedId", data.get("pubmedId"))
            if pubmed_id is not None:
                pubmed_id = str(pubmed_id)

            return GWASStudy(
                study_id=study_id,
                title=title,
                publication=publication,
                pub_date=pub_date,
                journal=journal,
                initial_sample_size=initial_sample_size,
                traits=traits,
                has_summary_stats=has_summary_stats,
                ftp_path=ftp_path,
                pubmed_id=pubmed_id,
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Failed to parse study: %s (%s)", data, exc)
            return None
