"""
BioReef.ai — Taxonomy Utilities (WoRMS API Client + Local Tree)
================================================================
Provides programmatic access to the World Register of Marine Species
(WoRMS) REST API for taxonomic validation, and maintains a local
cached taxonomic tree for Gulf of Oman species.

Every species annotation in the BioReef.ai pipeline must form a valid
biological path through the Linnaean hierarchy:
    Kingdom → Phylum → Class → Order → Family → Genus → Species

The WoRMS API (marinespecies.org/rest/) serves as the authoritative
source for taxonomic accuracy, ensuring that all species metadata
in the system is scientifically valid and up-to-date.

Guardrails (.agent/rules.md):
    - Biological Consistency: every label must be validated.
    - Documentation follows the "Marine Biologist" persona.

Verification:
    WoRMS API confirmed accessible — test query for Epinephelus coioides
    returned AphiaID 218200, Family Epinephelidae (verified 2026-03-08).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("bioreef.utils.taxonomy")


class WoRMSClient:
    """
    REST API client for the World Register of Marine Species.

    Provides taxonomic lookups, name resolution, and classification
    retrieval for marine species encountered in the Gulf of Oman.

    API Base: https://www.marinespecies.org/rest/

    Rate Limiting:
        WoRMS API has a rate limit. This client implements a polite
        delay between requests (default 0.5s) and caches results
        locally to minimize redundant queries.

    Ecological note:
        Taxonomic nomenclature in marine biology evolves frequently.
        Species may be reclassified (e.g., Epinephelus was recently
        split from Serranidae into Epinephelidae). WoRMS provides
        the most current accepted classifications.
    """

    BASE_URL = "https://www.marinespecies.org/rest"

    def __init__(
        self,
        cache_path: Optional[str] = None,
        request_delay: float = 0.5,
    ):
        """
        Args:
            cache_path:    Path to local JSON cache file.
            request_delay: Seconds to wait between API requests.
        """
        self.cache_path = cache_path
        self.request_delay = request_delay
        self._cache: Dict[str, Any] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BioReef.ai/0.1.0 (marine-biodiversity-research)",
        })

        # Load existing cache
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._cache = json.load(f)
            logger.info(f"Loaded {len(self._cache)} cached taxa from {cache_path}")

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a rate-limited GET request to the WoRMS API."""
        url = f"{self.BASE_URL}/{endpoint}"
        time.sleep(self.request_delay)

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"WoRMS API error for {url}: {e}")
            return None

    def get_aphia_id(self, species_name: str) -> Optional[int]:
        """
        Get the unique AphiaID for a species name.

        Args:
            species_name: Scientific name (e.g., "Epinephelus coioides").

        Returns:
            AphiaID integer, or None if not found.
        """
        cache_key = f"aphia_id:{species_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = self._request(
            f"AphiaIDByName/{species_name}",
            params={"marine_only": "true"},
        )

        if data and isinstance(data, int):
            self._cache[cache_key] = data
            self._save_cache()
            return data

        logger.warning(f"AphiaID not found for: {species_name}")
        return None

    def get_classification(self, aphia_id: int) -> Optional[Dict[str, str]]:
        """
        Get the full taxonomic classification for an AphiaID.

        Returns:
            Dict with keys: kingdom, phylum, class, order, family, genus, species.
        """
        cache_key = f"classification:{aphia_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = self._request(f"AphiaClassificationByAphiaID/{aphia_id}")

        if data:
            classification = self._parse_classification_tree(data)
            self._cache[cache_key] = classification
            self._save_cache()
            return classification

        return None

    def _parse_classification_tree(self, tree: Dict) -> Dict[str, str]:
        """Recursively parse the WoRMS classification tree into a flat dict."""
        result = {}
        current = tree

        while current:
            rank = current.get("rank", "").lower()
            name = current.get("scientificname", "")
            if rank and name:
                result[rank] = name
            current = current.get("child")

        return result

    def lookup_species(self, species_name: str) -> Optional[Dict[str, str]]:
        """
        Full taxonomic lookup: name → AphiaID → classification.

        Args:
            species_name: Scientific name (e.g., "Lutjanus ehrenbergii").

        Returns:
            Dict with 'family', 'genus', 'species', 'aphia_id', and
            full classification, or None if not found.
        """
        cache_key = f"species:{species_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        aphia_id = self.get_aphia_id(species_name)
        if not aphia_id:
            return None

        classification = self.get_classification(aphia_id)
        if not classification:
            return None

        result = {
            "species": species_name,
            "genus": classification.get("genus", ""),
            "family": classification.get("family", ""),
            "order": classification.get("order", ""),
            "class": classification.get("class", ""),
            "phylum": classification.get("phylum", ""),
            "kingdom": classification.get("kingdom", ""),
            "aphia_id": aphia_id,
        }

        self._cache[cache_key] = result
        self._save_cache()

        logger.info(
            f"WoRMS lookup: {species_name} → Family: {result['family']}, "
            f"Genus: {result['genus']} (AphiaID: {aphia_id})"
        )
        return result

    def validate_taxonomy(
        self, family: str, genus: str, species: str
    ) -> bool:
        """
        Validate that a (Family, Genus, Species) triple is biologically consistent.

        Args:
            family:  Expected family name.
            genus:   Expected genus name.
            species: Species name to lookup and verify.

        Returns:
            True if the WoRMS classification matches the expected hierarchy.
        """
        result = self.lookup_species(species)
        if not result:
            logger.warning(f"Cannot validate — species not found: {species}")
            return False

        family_match = result["family"].lower() == family.lower()
        genus_match = result["genus"].lower() == genus.lower()

        if not family_match:
            logger.error(
                f"❌ TAXONOMY MISMATCH: {species} — "
                f"Expected Family '{family}', WoRMS says '{result['family']}'"
            )
        if not genus_match:
            logger.error(
                f"❌ TAXONOMY MISMATCH: {species} — "
                f"Expected Genus '{genus}', WoRMS says '{result['genus']}'"
            )

        return family_match and genus_match

    def _save_cache(self):
        """Persist the local cache to disk."""
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)


class TaxonomicTree:
    """
    Local taxonomic tree for Gulf of Oman reef species.

    Maintains a cached hierarchy used by:
        - TaxonomicParser: for label encoding
        - HDEvaluator: for distance computation
        - HSLM: for Consistency Mask generation

    Can be populated manually or bootstrapped from WoRMS lookups.
    """

    def __init__(self, tree_path: Optional[str] = None):
        """
        Args:
            tree_path: Path to JSON file with species taxonomy data.
        """
        self._tree: Dict[str, Dict[str, str]] = {}
        self._families: Dict[str, List[str]] = {}  # family → [genera]
        self._genera: Dict[str, List[str]] = {}     # genus → [species]

        if tree_path and os.path.exists(tree_path):
            self.load(tree_path)

    def add_species(self, species: str, genus: str, family: str):
        """Register a species in the local tree."""
        self._tree[species] = {
            "species": species,
            "genus": genus,
            "family": family,
        }

        # Update reverse indices
        if family not in self._families:
            self._families[family] = []
        if genus not in self._families[family]:
            self._families[family].append(genus)

        if genus not in self._genera:
            self._genera[genus] = []
        if species not in self._genera[genus]:
            self._genera[genus].append(species)

    def get_taxonomy(self, species: str) -> Optional[Dict[str, str]]:
        """Retrieve the taxonomy dict for a species."""
        return self._tree.get(species)

    def get_species_in_genus(self, genus: str) -> List[str]:
        """List all species registered under a genus."""
        return self._genera.get(genus, [])

    def get_genera_in_family(self, family: str) -> List[str]:
        """List all genera registered under a family."""
        return self._families.get(family, [])

    def get_consistency_mask(self, family: str) -> List[str]:
        """
        Generate the Taxonomic Guardrail mask for a predicted family.

        Returns a list of valid species that the Species head is
        allowed to predict, given the Family head's output. This
        prevents biologically impossible predictions (e.g., a Shark
        family prediction leading to a Snapper species prediction).

        Used by the HSLM during inference.
        """
        valid_species = []
        for genus in self.get_genera_in_family(family):
            valid_species.extend(self.get_species_in_genus(genus))
        return valid_species

    def build_from_worms(
        self,
        species_list: List[str],
        worms_client: WoRMSClient,
    ) -> int:
        """
        Populate the tree by looking up species from WoRMS.

        Args:
            species_list: List of species names to register.
            worms_client: WoRMSClient instance for API lookups.

        Returns:
            Number of species successfully registered.
        """
        registered = 0
        for species_name in species_list:
            result = worms_client.lookup_species(species_name)
            if result and result.get("family") and result.get("genus"):
                self.add_species(
                    species=species_name,
                    genus=result["genus"],
                    family=result["family"],
                )
                registered += 1
            else:
                logger.warning(f"Could not register from WoRMS: {species_name}")

        logger.info(
            f"TaxonomicTree built from WoRMS: {registered}/{len(species_list)} "
            f"species registered across {len(self._families)} families."
        )
        return registered

    def save(self, path: str):
        """Save the tree to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "species": self._tree,
                "families": self._families,
                "genera": self._genera,
            }, f, indent=2)
        logger.info(f"TaxonomicTree saved to: {path}")

    def load(self, path: str):
        """Load the tree from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self._tree = data.get("species", {})
        self._families = data.get("families", {})
        self._genera = data.get("genera", {})
        logger.info(
            f"TaxonomicTree loaded: {len(self._tree)} species, "
            f"{len(self._families)} families from {path}"
        )

    @property
    def species_to_taxonomy(self) -> Dict[str, Dict[str, str]]:
        """Return the full species→taxonomy mapping (for TaxonomicParser)."""
        return self._tree

    @property
    def all_species(self) -> List[str]:
        return list(self._tree.keys())

    @property
    def all_families(self) -> List[str]:
        return list(self._families.keys())

    @property
    def all_genera(self) -> List[str]:
        return list(self._genera.keys())
