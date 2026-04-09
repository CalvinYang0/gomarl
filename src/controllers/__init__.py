REGISTRY = {}

from .basic_controller import BasicMAC
from .group_controller import NMAC as GroupMAC
from .graph_group_controller import GraphGroupMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["group_mac"] = GroupMAC
REGISTRY["graph_group_mac"] = GraphGroupMAC
