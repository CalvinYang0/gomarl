REGISTRY = {}

from .n_group_agent import GroupAgent
from .graph_group_agent import GraphGroupAgent

REGISTRY["n_group"] = GroupAgent
REGISTRY["graph_group"] = GraphGroupAgent
