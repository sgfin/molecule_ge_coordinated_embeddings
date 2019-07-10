import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss_WU(nn.Module):
    # Adapted from:  https://github.com/suruoxi/DistanceWeightedSampling/blob/master/model.py
    def __init__(self, margin=0.2, nu=0.0, beta=1.2, distance=nn.PairwiseDistance):
        super(TripletMarginLoss_WU, self).__init__()
        self._margin = margin
        self._nu = nu
        self.beta = beta
        self.distance = distance()

    def forward(self, anchors, positives, negatives):
        d_ap = self.distance(positives, anchors)
        d_an = self.distance(negatives, anchors)

        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

        loss = (torch.sum(pos_loss + neg_loss)) / pair_cnt
        return loss, pair_cnt

class QuadrupletMarginLoss(nn.Module):
    # Idea here is to do margin loss with anchor_ge vs anchor_chem/non_match_chem
    # and additionally do the margin loss with anchor_chem vs anchor_ge/non_match_ge
    # Each one gets an output and you backprop over both
    # Could temporarily call this a double-margin loss
    def __init__(self,  *args, **kwargs):
        super(QuadrupletMarginLoss, self).__init__()
        self.triplet_margin = TripletMarginLoss_WU(*args, **kwargs)

    def forward(self, anchor_ge, non_match_ge, anchor_chem, non_match_chem):
        loss_ge, pair_cnt_ge = self.triplet_margin(anchor_ge, anchor_chem, non_match_chem)
        loss_chem, pair_cnt_chem = self.triplet_margin(anchor_chem, anchor_ge, non_match_ge)

        loss_total = loss_ge + loss_chem

        return loss_total, (pair_cnt_ge, pair_cnt_chem)

class QuintupletMarginLoss(nn.Module):
    # margin loss with anchor_ge vs anchor_chem/non_match_chem
    # margin loss with anchor_chem vs anchor_ge/non_match_ge
    # margin loss with anchor_ge vs match_ge/non_match_ge
    def __init__(self,  *args, **kwargs):
        super(QuintupletMarginLoss, self).__init__()
        self.triplet_margin = TripletMarginLoss_WU(*args, **kwargs)

    def forward(self, anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem):
        loss_ge_first, pair_cnt_ge_first = self.triplet_margin(anchor_ge, anchor_chem, non_match_chem)
        loss_chem_first, pair_cnt_chem_first = self.triplet_margin(anchor_chem, anchor_ge, non_match_ge)
        loss_all_ge, pair_cnt_all_get = self.triplet_margin(anchor_ge, match_ge, non_match_ge)

        loss_total = loss_ge_first + loss_chem_first + loss_all_ge

        return loss_total, (pair_cnt_ge_first, pair_cnt_chem_first, pair_cnt_all_get)