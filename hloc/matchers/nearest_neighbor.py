import torch

from ..utils.base_model import BaseModel


def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        pass

    def _forward(self, data):
        sim = torch.einsum(
            'bdn,bdm->bnm', data['descriptors0'], data['descriptors1'])
        matches0, scores0 = find_nn(
            sim, self.conf['ratio_threshold'], self.conf['distance_threshold'])
        matches1, scores1 = find_nn(
            sim.transpose(1, 2), self.conf['ratio_threshold'],
            self.conf['distance_threshold'])
        if self.conf['do_mutual_check']:
            matches1, scores1 = find_nn(
                sim.transpose(1, 2), self.conf['ratio_threshold'],
                self.conf['distance_threshold'])
            matches0 = mutual_check(matches0, matches1)
        return {
            'matches0': matches0,
            'matching_scores0': scores0,
        }
