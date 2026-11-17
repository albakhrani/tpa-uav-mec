"""Common-random-number streams.

Every source of randomness has its own stream, derived from
SeedSequence([seed, STREAM_ID, block]). Environment streams (world, arrivals,
channel, mobility, loss) never depend on the arm or the policy, so within a
seed every arm sees identical arrivals and channels. Blocks separate the
training, validation and test traces of the same seed.

The amount drawn from an environment stream per slot must not depend on the
policy (e.g. draw the full N x M fading matrix every slot).
"""
from __future__ import annotations

import numpy as np

STREAMS = {
    "world": 1,      # world layout (block-independent: same world for train/val/test)
    "arrivals": 2,   # arrival counts + task attributes + per-task loss uniforms
    "channel": 3,    # small-scale fading xi_ij(t)
    "mobility": 4,   # Gauss-Markov device motion
    "loss": 5,       # spare (packet-loss uniforms are drawn per task from "arrivals")
    "net_init": 6,   # network initialisation (torch seed derived from it)
    "policy": 7,     # action sampling
    "replay": 8,     # replay minibatch sampling
    "a14": 9,        # A14 displacement draws
    "tuning": 10,    # hyperparameter sampler
    "a15": 11,       # A15 key-shuffle draws pi(t)
}

BLOCKS = {"train": 0, "val": 1, "test": 2, "calib": 3}


class Streams:
    """Named numpy Generators for one (seed, block)."""

    def __init__(self, seed: int, block: str = "train", test_run: int = 0):
        self.seed, self.block, self.test_run = int(seed), block, int(test_run)
        b = BLOCKS[block] * 1000 + self.test_run
        self._gens: dict[str, np.random.Generator] = {}
        for name, sid in STREAMS.items():
            # the world is shared by all blocks of a seed
            key = [self.seed, sid, 0 if name == "world" else b]
            self._gens[name] = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))

    def __getattr__(self, name: str) -> np.random.Generator:
        gens = self.__dict__.get("_gens")
        if gens is not None and name in gens:
            return gens[name]
        raise AttributeError(name)

    def torch_seed(self) -> int:
        """Deterministic 63-bit torch seed from the net_init stream (does not consume it)."""
        ss = np.random.SeedSequence([self.seed, STREAMS["net_init"], 0])
        return int(ss.generate_state(1, dtype=np.uint64)[0] & ((1 << 63) - 1))

    # ---- checkpoint support ----
    def get_state(self) -> dict:
        return {n: g.bit_generator.state for n, g in self._gens.items()}

    def set_state(self, state: dict) -> None:
        for n, s in state.items():
            self._gens[n].bit_generator.state = s
