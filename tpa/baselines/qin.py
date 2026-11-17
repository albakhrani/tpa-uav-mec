"""B_QIN: static-hover restriction of Qin et al., "Joint Resource Allocation and Trajectory
Optimization for Multi-UAV-Assisted Multi-Access Mobile Edge Computing", IEEE WCL 10(7), 2021.

Published method: block-coordinate (alternating) optimisation of user association /
bandwidth and computation resource allocation, and of the UAV trajectories, for a delay
objective with multiple radio access.
Restriction used here: the trajectory block is removed (fixed hover), bandwidth
is not allocated (uplink rates follow the access model cfg.channel.access, as for every
arm), so only the resource block remains, solved per slot:
  * assignment: earliest predicted finish -- each task (FIFO window) to the feasible UAV
    minimising the predicted T_total (Eq. 13) with the share f_est = F0_j / (n_j + 1)
    it would get if UAV j's slot-start residual F0_j were split over its n_j tasks
    already assigned this slot plus this one (f_est floored at f_req);
  * allocation: the closed-form minimiser of sum_i c_i phi_i / f_i s.t. sum_i f_i <= F0_j
    (Lagrangian/KKT): f_i = F0_j sqrt(c_i phi_i) / sum_k sqrt(c_k phi_k) over the tasks
    assigned to j, then f = clip(max(f_i, f_req), 0, F_res).
No learning; deterministic given the trace.
"""
from __future__ import annotations

import math

import numpy as np

from ..env.queueing import predict_total_time
from .zhou import _RuleController


class QinController(_RuleController):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._n = None
        self._t = None

    def act(self, obs, label, train):
        self._n = np.zeros(self.ex.M, dtype=np.int64)      # tasks assigned to j so far (pass 1)
        return super().act(obs, label, train)

    def choose(self, obs, i, k, cands):
        task = obs.queue[i]
        F0 = np.asarray(obs.residual_f, dtype=np.float64)
        eta = obs.eta
        best, best_t = -1, math.inf
        for j, f_req, f_hi, R in cands:
            f_est = max(F0[j] / (self._n[j] + 1), f_req)
            tp = predict_total_time(task.size_bits, task.cycles, task.complexity, R, f_est,
                                    float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                                    self.ex.t_mem, self.cfg)
            if tp < best_t:
                best, best_t = j, tp
        if best >= 0:
            self._n[best] += 1
        return best

    def alloc(self, obs, j, items, F0):
        w = np.array([math.sqrt(obs.queue[i].cycles * obs.queue[i].complexity)
                      for (i, _, _, _) in items])
        return list(F0 * w / w.sum())
