"""Tests for the GWAS Catalog REST client.

Uses the ``responses`` library to mock HTTP requests and verify that the
client correctly parses HAL+JSON payloads into domain objects.
"""

import responses

from gwas_meta.gwas_client.models import GWASStudy, GWASTrait
from gwas_meta.gwas_client.rest_client import GWASCatalogClient

BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

def _trait_hal(short_form: str, trait: str, uri: str) -> dict:
    """Build a single EFO-trait object in HAL format."""
    return {"shortForm": short_form, "trait": trait, "uri": uri}


def _study_hal(
    accession: str = "GCST000001",
    title: str = "A genome-wide study",
    author: str = "Smith J",
    pub_date: str = "2023-01-15",
    journal: str = "Nature",
    sample_size: str = "5,000 European ancestry",
    full_pvalue_set: bool = True,
    efo_traits: list[dict] | None = None,
) -> dict:
    """Build a single study object in HAL format."""
    study: dict = {
        "accessionId": accession,
        "publicationInfo": {
            "title": title,
            "publicationDate": pub_date,
            "author": {"fullname": author},
            "publication": journal,
        },
        "initialSampleSize": sample_size,
        "fullPvalueSet": full_pvalue_set,
    }
    if efo_traits:
        study["_embedded"] = {"efoTraits": efo_traits}
    return study


def _hal_page(resource_key: str, items: list[dict], next_href: str | None = None) -> dict:
    """Wrap a list of items in a HAL page envelope."""
    page: dict = {"_embedded": {resource_key: items}, "_links": {}}
    if next_href:
        page["_links"]["next"] = {"href": next_href}
    return page


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestSearchTraits:
    @responses.activate
    def test_returns_parsed_traits(self) -> None:
        url = f"{BASE_URL}/efoTraits/search/findByEfoTrait"
        body = _hal_page("efoTraits", [
            _trait_hal("EFO_0000270", "asthma", "http://www.ebi.ac.uk/efo/EFO_0000270"),
            _trait_hal("EFO_0000275", "atopic asthma", "http://www.ebi.ac.uk/efo/EFO_0000275"),
        ])
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        traits = client.search_traits("asthma")

        assert len(traits) == 2
        assert all(isinstance(t, GWASTrait) for t in traits)
        assert traits[0].efo_id == "EFO_0000270"
        assert traits[0].trait_name == "asthma"
        assert traits[1].efo_id == "EFO_0000275"


class TestSearchByEfo:
    @responses.activate
    def test_returns_studies_for_efo(self) -> None:
        url = f"{BASE_URL}/studies/search/findByEfoUri"
        body = _hal_page("studies", [
            _study_hal(accession="GCST000001", title="Study A"),
            _study_hal(accession="GCST000002", title="Study B", full_pvalue_set=False),
        ])
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        studies = client.search_by_efo("EFO_0000270")

        assert len(studies) == 2
        assert all(isinstance(s, GWASStudy) for s in studies)
        assert studies[0].study_id == "GCST000001"
        assert studies[0].has_summary_stats is True
        assert studies[1].has_summary_stats is False

    @responses.activate
    def test_accepts_full_uri(self) -> None:
        url = f"{BASE_URL}/studies/search/findByEfoUri"
        body = _hal_page("studies", [_study_hal()])
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        studies = client.search_by_efo("http://www.ebi.ac.uk/efo/EFO_0000270")

        assert len(studies) == 1


class TestSearchStudies:
    @responses.activate
    def test_search_by_disease_trait(self) -> None:
        url = f"{BASE_URL}/studies/search/findByDiseaseTrait"
        body = _hal_page("studies", [
            _study_hal(accession="GCST000010", title="Asthma GWAS"),
        ])
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        studies = client.search_studies("asthma")

        assert len(studies) == 1
        assert studies[0].study_id == "GCST000010"


class TestGetStudy:
    @responses.activate
    def test_returns_single_study(self) -> None:
        study_id = "GCST000001"
        url = f"{BASE_URL}/studies/{study_id}"
        body = _study_hal(accession=study_id, title="My GWAS")
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        study = client.get_study(study_id)

        assert isinstance(study, GWASStudy)
        assert study.study_id == study_id
        assert study.title == "My GWAS"


class TestPagination:
    @responses.activate
    def test_follows_next_links(self) -> None:
        url = f"{BASE_URL}/efoTraits/search/findByEfoTrait"
        page2_url = f"{BASE_URL}/efoTraits/search/findByEfoTrait?page=1"

        page1 = _hal_page(
            "efoTraits",
            [_trait_hal("EFO_0000001", "trait1", "http://efo/1")],
            next_href=page2_url,
        )
        page2 = _hal_page(
            "efoTraits",
            [_trait_hal("EFO_0000002", "trait2", "http://efo/2")],
            next_href=None,
        )
        responses.add(responses.GET, url, json=page1, status=200)
        responses.add(responses.GET, page2_url, json=page2, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        traits = client.search_traits("trait")

        assert len(traits) == 2
        assert traits[0].efo_id == "EFO_0000001"
        assert traits[1].efo_id == "EFO_0000002"
        # Two HTTP requests should have been issued
        assert len(responses.calls) == 2

    @responses.activate
    def test_stops_at_max_pages(self) -> None:
        url = f"{BASE_URL}/efoTraits/search/findByEfoTrait"
        # Every page points to itself -> infinite loop without max_pages guard
        body = _hal_page(
            "efoTraits",
            [_trait_hal("EFO_0000001", "t", "http://efo/1")],
            next_href=url,
        )
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        # Default max_pages is 10, so we should get 10 pages of 1 trait each
        traits = client.search_traits("t")
        assert len(traits) == 10
        assert len(responses.calls) == 10


class TestStudyWithEmbeddedTraits:
    @responses.activate
    def test_parses_embedded_efo_traits(self) -> None:
        url = f"{BASE_URL}/studies/GCST000001"
        efo = _trait_hal("EFO_0000270", "asthma", "http://www.ebi.ac.uk/efo/EFO_0000270")
        body = _study_hal(accession="GCST000001", efo_traits=[efo])
        responses.add(responses.GET, url, json=body, status=200)

        client = GWASCatalogClient(rate_limit_delay=0)
        study = client.get_study("GCST000001")

        assert len(study.traits) == 1
        assert study.traits[0].efo_id == "EFO_0000270"
