from .base import BaseComparator
from .ede import EDEComparator
from .stdde import StdDEComparator
from .alns_ms import ALNSMSComparator
from .hgs_ms import HGSMSComparator
from .ils_ms import ILSMSComparator
from .ablations import AblationComparator

REGISTRY = {
    "EDE": EDEComparator,
    "StdDE": StdDEComparator,
    "ALNS_MS": ALNSMSComparator,
    "HGS_MS": HGSMSComparator,
    "ILS_MS": ILSMSComparator,
    "A1_NoSeed": AblationComparator,
    "A2_NoJDE": AblationComparator,
    "A3_NoLNS": AblationComparator,
}
