import torch
import torch.nn as nn

class LambdaLoss(nn.Module):
    def __init__(self):
        super(LambdaLoss, self).__init__()

    def forward(self, preds, targets):
        batch_size = preds.size(0)
        num_items = preds.size(1)


        sorted_preds, indices = torch.sort(preds, descending=True, dim=1)
        sorted_targets = torch.gather(targets, dim=1, index=indices)


        diffs = sorted_preds.unsqueeze(2) - sorted_preds.unsqueeze(1)


        pairwise_loss = torch.sigmoid(-diffs)


        def delta_ndcg(i, j, ranks, targets):
            gain_diff = targets[:, i] - targets[:, j]
            discount_diff = torch.log2(ranks[:, j].float().clone().detach() + 2) - torch.log2(ranks[:, i].float().clone().detach() + 2)
            delta = torch.abs(gain_diff * discount_diff)
            return delta

        deltas = torch.zeros((batch_size, num_items, num_items), dtype=torch.float32, device=preds.device)
        ranks = torch.arange(1, num_items + 1, device=preds.device).unsqueeze(0).expand(batch_size, num_items)
        for i in range(num_items):
            for j in range(num_items):
                deltas[:, i, j] = delta_ndcg(i, j, ranks, sorted_targets)


        loss = (pairwise_loss * deltas).sum() / batch_size #Dividing by batch size for correct scaling
        return loss