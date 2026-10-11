REGISTRY = {}

from .clean_hyper_agent import CleanHyperAgent
from .n_group_agent import GroupAgent
from .slot_group_hyper_agent import SlotGroupHyperAgent

REGISTRY["clean_hyper"] = CleanHyperAgent
REGISTRY["n_group"] = GroupAgent
REGISTRY["slot_group_hyper"] = SlotGroupHyperAgent
