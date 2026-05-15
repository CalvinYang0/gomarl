REGISTRY = {}

from .basic_controller import BasicMAC
from .clean_controller import CleanMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["clean_mac"] = CleanMAC
