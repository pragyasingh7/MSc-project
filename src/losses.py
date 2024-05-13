import torch 
from torch import nn

from src.utils import compute_topological_measures


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, predicted):
        target_ = target - torch.mean(target)
        pred_ = predicted - torch.mean(predicted)
        corr = torch.sum(target_ * pred_) / (torch.sqrt(torch.sum(target_ ** 2)) * torch.sqrt(torch.sum(pred_ ** 2)))
        return 1 - corr


class TopoMetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, target, predicted, return_components=False):
        # Compute topological measures
        # cc_target, bc_target, ec_target = compute_topological_measures(target)
        # cc_pred, bc_pred, ec_pred = compute_topological_measures(predicted)

        ec_target = compute_topological_measures(target)
        ec_pred = compute_topological_measures(predicted)

        # Compute topological loss
        # cc_loss = self.l1_loss(cc_target, cc_pred)
        # bc_loss = self.l1_loss(bc_target, bc_pred)
        ec_loss = self.l1_loss(ec_target, ec_pred)

        # Total loss
        # total_loss = cc_loss + bc_loss + ec_loss
        total_loss = ec_loss

        if return_components:
            # return total_loss, cc_loss, bc_loss, ec_loss
            return total_loss, ec_loss

        return total_loss


class GTPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.topo_loss = TopoMetricLoss()
        self.pearson_loss = PearsonCorrelationLoss()

    def forward(self, target, predicted):
        # L1 loss
        l1_loss = self.l1_loss(target, predicted)

        # Topological loss
        topo_loss = self.topo_loss(target, predicted)

        # Pearson correlation loss
        pearson_loss = self.pearson_loss(target, predicted)

        # Total loss
        loss = l1_loss + topo_loss + pearson_loss

        return loss