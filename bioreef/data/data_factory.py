"""
BioReef.ai — Data Factory (compatibility shim)
===============================================
The Stage-1 preprocessing pipeline was split into focused modules:

    Step 2  Spectral restoration  -> bioreef.data.restoration   (WaterNetRestorer)
    Step 3  Context cropping      -> bioreef.data.context       (ContextHarvester)
    Step 1  Metadata parse/filter -> bioreef.data.taxonomy_parse(TaxonomicParser)
    Step 5  Augmentation          -> bioreef.data.augmentation  (MarineAugmentor)
            Dataset assembly      -> bioreef.data.dataset       (BioReefDataset)

This module now ONLY re-exports those names so existing
`from bioreef.data.data_factory import X` imports keep working unchanged. New
code should import from the specific module above.
"""

from bioreef.data.restoration import (   # noqa: F401
    WaterNetRestorer,
    _wn_white_balance,
    _wn_gamma,
    _wn_histeq,
)
from bioreef.data.context import ContextHarvester           # noqa: F401
from bioreef.data.taxonomy_parse import TaxonomicParser     # noqa: F401
from bioreef.data.augmentation import MarineAugmentor       # noqa: F401
from bioreef.data.dataset import BioReefDataset             # noqa: F401

__all__ = [
    "WaterNetRestorer", "ContextHarvester", "TaxonomicParser",
    "MarineAugmentor", "BioReefDataset",
    "_wn_white_balance", "_wn_gamma", "_wn_histeq",
]
