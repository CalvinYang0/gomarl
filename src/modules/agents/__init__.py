REGISTRY = {}

from .clean_hyper_agent import CleanHyperAgent
from .n_group_agent import GroupAgent

REGISTRY["clean_hyper"] = CleanHyperAgent
REGISTRY["n_group"] = GroupAgent
