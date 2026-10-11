REGISTRY = {}

from .basic_controller import BasicMAC
from .clean_controller import CleanMAC
from .group_controller import NMAC as GroupMAC
from .slot_group_controller import SlotGroupMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["clean_mac"] = CleanMAC
REGISTRY["group_mac"] = GroupMAC
REGISTRY["slot_group_mac"] = SlotGroupMAC
