import torch
from scipy.optimize import linear_sum_assignment

from tracker.qdtrack.builder import TRACKERS


@TRACKERS.register_module()
class MiniVISTracker:
    def __init__(
            self
    ):
        self.memory = None

    def match_from_embds(self, tgt_embds, cur_embds):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def match(self, cur_embds):
        if self.memory is None:
            self.memory = cur_embds
            return torch.arange(len(cur_embds), dtype=torch.long, device=cur_embds.device)

        cur_embds = cur_embds / cur_embds.norm(dim=1, keepdim=True)
        tgt_embds = self.memory / self.memory.norm(dim=1, keepdim=True)

        sim = cur_embds @ tgt_embds.t()
        cost = 1 - sim

        C = 1.0 * cost
        C = C.cpu()

        new_ids = - torch.ones_like(cur_embds[:, 0], dtype=torch.long)
        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = (torch.tensor(indices[0], device=cur_embds.device), torch.tensor(indices[1], device=cur_embds.device))
        if len(cur_embds) > len(tgt_embds):
            indices_already = indices[1]
            indices_not = []
            for i in range(len(new_ids)):
                if i not in indices_already:
                    indices_not.append(i)
            indices_not = torch.tensor(indices_not, device=cur_embds.device)
            new_ids[indices[1]] = indices[0]
            new_ids[indices_not] = torch.arange(len(indices_not), device=cur_embds.device) + len(self.memory)
            self.memory = torch.cat([self.memory, cur_embds[indices_not]])
        else:
            new_ids[indices[1]] = indices[0]
        assert -1 not in new_ids
        return new_ids
