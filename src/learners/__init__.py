from .clean_learner import CleanLearner
from .group_learner import GROUPLearner

REGISTRY = {}

REGISTRY["clean_learner"] = CleanLearner
REGISTRY["group_learner"] = GROUPLearner
