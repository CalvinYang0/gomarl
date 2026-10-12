from .clean_learner import CleanLearner
from .group_learner import GROUPLearner
from .slot_group_learner import SlotGroupLearner
from .mixer_input_learner import MixerInputLearner

REGISTRY = {}

REGISTRY["clean_learner"] = CleanLearner
REGISTRY["group_learner"] = GROUPLearner
REGISTRY["slot_group_learner"] = SlotGroupLearner
REGISTRY["mixer_input_learner"] = MixerInputLearner
