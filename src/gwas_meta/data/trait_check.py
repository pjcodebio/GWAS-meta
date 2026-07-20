"""Pre-flight EFO-trait compatibility check across selected studies.

Meta-analysing effect sizes from studies of *different* traits produces
uninterpretable output (β from a log-OR HbA1c study pooled with β from a
standardized-BMI study is mathematically well-defined but scientifically
meaningless). Step 3 searches the GWAS Catalog by EFO, so in the normal
flow all selected studies share at least one EFO trait. But nothing
enforces that at meta-analysis time — the user can select studies from
different Catalog searches, or the LLM-generated criteria can span
multiple traits. This module surfaces the mismatch as a visible warning
without blocking the run (legitimate cross-phenotype meta-analyses of
related sub-traits exist).

Notes on scope:
- EFO IDs are compared exactly; the hierarchy (parent/child terms) is
  not consulted, so two child terms of a shared parent would still
  trigger the warning.
- Trait *names* are ignored — free-text names are unreliable across
  Catalog releases; only ``efo_id`` is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraitMismatchFinding:
    """Result of the cross-study trait compatibility check.

    Parameters
    ----------
    shared_efo_ids:
        EFO IDs present in *every* study's trait list (the intersection).
        Empty when studies do not share any trait.
    per_study:
        Mapping of ``study_id`` → list of ``(efo_id, trait_name)`` pairs
        for that study, preserved for display in the warning.
    """

    shared_efo_ids: set[str]
    per_study: dict[str, list[tuple[str, str]]]

    @property
    def is_mismatch(self) -> bool:
        """True when studies do not share any EFO trait."""
        return len(self.shared_efo_ids) == 0 and len(self.per_study) >= 2


def check_trait_compatibility(studies: "list") -> TraitMismatchFinding:
    """Check whether all studies share at least one EFO trait.

    Parameters
    ----------
    studies:
        Iterable of objects with ``study_id`` and ``traits`` attributes
        (i.e. :class:`gwas_meta.gwas_client.models.GWASStudy`, where
        each entry in ``traits`` has ``efo_id`` and ``trait_name``).

    Returns
    -------
    TraitMismatchFinding
        Aggregated view; call ``.is_mismatch`` to decide whether to warn.
    """
    per_study: dict[str, list[tuple[str, str]]] = {}
    efo_sets: list[set[str]] = []

    for st in studies:
        traits = getattr(st, "traits", []) or []
        pairs = [
            (getattr(t, "efo_id", ""), getattr(t, "trait_name", ""))
            for t in traits
            if getattr(t, "efo_id", "")
        ]
        per_study[st.study_id] = pairs
        efo_sets.append({efo for efo, _ in pairs})

    if not efo_sets:
        shared: set[str] = set()
    else:
        shared = set.intersection(*efo_sets) if all(efo_sets) else set()

    return TraitMismatchFinding(shared_efo_ids=shared, per_study=per_study)
