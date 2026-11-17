"""Published-method baselines (B_ZHOU, B_QIN, B_HUANG, B_SONG) through the shared
pattern-free executor, and the controls A10 (LSTM actor-critic), A11 (frame stack)
and A12 (zone x phase lookup). All implement the TPAController protocol:
reset_segment(t0), start_episode(), act(obs, label, train) -> (Decision, mem_energy_j),
observe(obs, dec, outcome, train) -> info, eval_clone(streams), get_state()/set_state()."""
from .executor import Executor, fifo_window
from .zhou import ZhouController
from .qin import QinController
from .huang import HuangController
from .song import SongController
from .lookup import LookupController
from .lstm_ac import LSTMController, param_counts
from .framestack import FrameStackController

CONTROLLERS = {
    "zhou": ZhouController,
    "qin": QinController,
    "huang": HuangController,
    "song": SongController,
    "lookup": LookupController,
    "lstm": LSTMController,
    "framestack": FrameStackController,
}

__all__ = ["Executor", "fifo_window", "CONTROLLERS", "param_counts", "ZhouController",
           "QinController", "HuangController", "SongController", "LookupController",
           "LSTMController", "FrameStackController"]
