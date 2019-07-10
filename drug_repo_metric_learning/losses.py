import torch
import torch.nn as nn


class TripletMarginLoss_WU(nn.Module):
    # Adapted from:  https://github.com/suruoxi/DistanceWeightedSampling/blob/master/model.py
    def __init__(self, margin=0.2, nu=0.0, beta=1.2, distance=nn.PairwiseDistance,
                 percent_correct=True):
        super().__init__()
        self._margin = margin
        self._nu = nu
        self.beta = beta
        self.distance = distance()
        self.percent_correct = percent_correct

    def forward(self, anchors, positives, negatives):
        d_ap = self.distance(positives, anchors)
        d_an = self.distance(negatives, anchors)

        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))
        loss = (torch.sum(pos_loss + neg_loss)) / pair_cnt

        if self.percent_correct:
            perc_correct = torch.sum(d_an > d_ap).item() / len(d_an) * 100
            return loss, perc_correct
        else:
            return loss


class QuadrupletMarginLoss(nn.Module):
    # Idea here is to do margin loss with anchor_ge vs anchor_chem/non_match_chem
    # and additionally do the margin loss with anchor_chem vs anchor_ge/non_match_ge
    # Each one gets an output and you backprop over both
    # Could temporarily call this a double-margin loss
    def __init__(self, percent_correct=True, *args, **kwargs):
        super().__init__()
        self.percent_correct = percent_correct
        self.triplet_margin = TripletMarginLoss_WU(percent_correct=percent_correct, *args, **kwargs)

    def forward(self, anchor_ge, non_match_ge, anchor_chem, non_match_chem):
        res_ge = self.triplet_margin(anchor_ge, anchor_chem, non_match_chem)
        res_chem = self.triplet_margin(anchor_chem, anchor_ge, non_match_ge)

        if self.percent_correct:
            loss_ge, perc_correct_ge = res_ge
            loss_chem, perc_correct_chem = res_chem
        else:
            loss_ge = res_ge
            loss_chem = res_chem

        loss_total = loss_ge + loss_chem

        if self.percent_correct:
            return loss_total, (perc_correct_ge, perc_correct_chem)
        else:
            return loss_total


class QuintupletMarginLoss(nn.Module):
    def __init__(self, percent_correct=True, *args, **kwargs):
        super().__init__()
        self.percent_correct = percent_correct
        self.triplet_margin = TripletMarginLoss_WU(percent_correct=percent_correct, *args, **kwargs)

    def forward(self, anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem):
        res_ge_first = self.triplet_margin(anchor_ge, anchor_chem, non_match_chem)
        res_chem_first = self.triplet_margin(anchor_chem, anchor_ge, non_match_ge)
        res_all_ge = self.triplet_margin(anchor_ge, match_ge, non_match_ge)

        if self.percent_correct:
            loss_ge_first, perc_correct_ge_first = res_ge_first
            loss_chem_first, perc_correct_chem_first = res_chem_first
            loss_all_ge, perc_correct_all_ge = res_all_ge
        else:
            loss_ge_first = res_ge_first
            loss_chem_first = res_chem_first
            loss_all_ge = res_all_ge

        loss_total = loss_ge_first + loss_chem_first + loss_all_ge

        if self.percent_correct:
            return loss_total, (perc_correct_ge_first, perc_correct_chem_first, perc_correct_all_ge)
        else:
            return loss_total
