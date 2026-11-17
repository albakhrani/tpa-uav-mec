"""Every baseline arm end to end through tpa.run.trainer.run on the stage-1 path
(phase "tune": training, validation checkpoints, rolling checkpoints, end-of-run val_full
evaluation).

The test runs the baselines through the trainer past the first learner update, and B_SONG past
its first antithetic pair inside one memory segment, which covers two failure modes:
  * B_HUANG: trainer._ep_row reading u["alpha"] from the DDPG update dict (KeyError);
  * B_SONG: the next even episode after a pair starting one window later than the Simulator
    expects ("slots must be consecutive"), and an ES update log with none of the SAC keys.
The tiny config of tests/_sys_helpers.py holds one pair per segment (segment_slots = 2 t_ep),
which hides the B_SONG case; here a segment holds 12 episodes, as in config/default.toml.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _sys_helpers import row, tiny_cfg  # noqa: E402
from tpa.env.simulator import Simulator  # noqa: E402
from tpa.run import trainer as TR  # noqa: E402
from tpa.run.logio import read_all  # noqa: E402

T_EP, EPISODES = 40, 6
E2E = {"run.episodes": EPISODES, "run.t_ep": T_EP, "run.segment_slots": 12 * T_EP,   # 12 episodes per segment
       "baselines.es_pairs": 2}                                                     # ES iteration = 4 episodes


@pytest.mark.parametrize("arm", ["B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"])
def test_baseline_runs_through_trainer_on_stage1_path(arm, tmp_path, monkeypatch):
    first_slot = {}
    orig = Simulator.begin_slot

    def spy(self, t):
        first_slot.setdefault(TR._CUR_EP[0], t)
        return orig(self, t)

    monkeypatch.setattr(Simulator, "begin_slot", spy)
    cfg = tiny_cfg(**E2E)
    r = row(tmp_path, arm, phase="tune")
    res = TR.run(r, cfg)
    d = Path(r["run_dir"])
    assert res["episodes"] == EPISODES
    t = read_all(d / "train").sort_values("episode")
    assert list(t["episode"]) == list(range(EPISODES))
    assert all(int(s) == T_EP for s in t["slots"])
    assert (d / "RESULT.json").exists() and json.loads((d / "RESULT.json").read_text())["test_rows"] > 0
    assert len(read_all(d / "val_full")) > 0
    starts = [first_slot[e] for e in range(EPISODES)]
    if arm == "B_SONG":
        # the odd episode replays the even window; the odd window is skipped
        assert starts == [0, 0, 2 * T_EP, 2 * T_EP, 4 * T_EP, 4 * T_EP]
        # the first ES iteration (episodes 0-3) closes at the start of episode 4: one update
        assert int(t["n_updates"].sum()) == 1 and int(t["n_updates"].iloc[4]) == 1
        assert math.isnan(t["alpha"].iloc[4]) and math.isnan(t["critic_loss"].iloc[4])
    else:
        assert starts == [e * T_EP for e in range(EPISODES)]
    if arm == "B_HUANG":
        assert int(t["n_updates"].sum()) > 0                       # DDPG updates were logged
        upd = t[t["n_updates"] > 0]
        assert np.isnan(upd["alpha"]).all() and np.isfinite(upd["critic_loss"]).all()
    if arm in ("B_ZHOU", "B_QIN"):
        assert int(t["n_updates"].sum()) == 0 and np.isnan(t["alpha"]).all()
