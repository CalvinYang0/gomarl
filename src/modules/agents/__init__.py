REGISTRY = {}

from .clean_hyper_agent import CleanHyperAgent
from .n_group_agent import GroupAgent
from .slot_group_hyper_agent import SlotGroupHyperAgent
from third_party.pymarl2_qmix.agent import NRNNAgent as OfficialQMIXAgent

REGISTRY["clean_hyper"] = CleanHyperAgent
REGISTRY["n_group"] = GroupAgent
REGISTRY["slot_group_hyper"] = SlotGroupHyperAgent
REGISTRY["official_pymarl2_rnn"] = OfficialQMIXAgent
