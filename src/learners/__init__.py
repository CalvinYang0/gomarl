from .group_learner import GROUPLearner
from .graph_group_learner import GraphGROUPLearner

REGISTRY = {}

REGISTRY["group_learner"] = GROUPLearner
REGISTRY["graph_group_learner"] = GraphGROUPLearner
