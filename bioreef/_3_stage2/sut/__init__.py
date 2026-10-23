"""SU-T tracker (Li et al., "When Trackers Date Fish", AAAI 2026), MIT license (see LICENSE).

Vendored from github.com/vranlee/SU-T @ 479be89 (2026-07-30): trackers/SUT_UKF.py -> sut_ukf.py,
trackers/association.py and trackers/UKF.py unchanged. Only local change: one numpy-2 fix in sut_ukf.py marked
"BioReef patch". Needs filterpy + lapx.
"""
from .sut_ukf import SU_T
