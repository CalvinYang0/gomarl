from .clean_learner import CleanLearner
from .group_learner import GROUPLearner
from .slot_group_learner import SlotGroupLearner
from .mixer_input_learner import MixerInputLearner
from .onefill_gate_learner import OneFillGateLearner
from .official_qmix_learner import OfficialQMIXLearner
from .qtot_composition_learner import QTotCompositionLearner
from .soft_onefill_gate_learner import SoftOneFillGateLearner
from .mixer_state_refresh_learner import MixerStateRefreshLearner

REGISTRY = {}

REGISTRY["clean_learner"] = CleanLearner
REGISTRY["group_learner"] = GROUPLearner
REGISTRY["slot_group_learner"] = SlotGroupLearner
REGISTRY["mixer_input_learner"] = MixerInputLearner
REGISTRY["onefill_gate_learner"] = OneFillGateLearner
REGISTRY["official_pymarl2_qmix"] = OfficialQMIXLearner
REGISTRY["qtot_composition_learner"] = QTotCompositionLearner
REGISTRY["soft_onefill_gate_learner"] = SoftOneFillGateLearner
REGISTRY["mixer_state_refresh_learner"] = MixerStateRefreshLearner
