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


################################

class CCALoss(nn.Module):
    # Adapted from https://github.com/usc-sail/mica-deep-mcca/blob/master/objectives.py
    def __init__(self, use_all_singular_values=True, k=50):
        super().__init__()
        self.r1 = 1e-4
        self.r2 = 1e-4
        self.eps = 1e-12
        self.use_all_singular_values = use_all_singular_values
        if not self.use_all_singular_values:
            self.k = k

    def forward(self, ge, chem):
        ge = ge.t()
        chem = chem.t()
        o1 = o2 = ge.shape[0]
        m = ge.shape[1]

        ge_bar = ge - (1.0 / m) * torch.mm(ge, torch.ones([m, m]).cuda())
        chem_bar = chem - (1.0 / m) * torch.mm(chem, torch.ones([m, m]).cuda())

        SigmaHat12 = (1.0 / (m - 1)) * torch.mm(ge_bar, chem_bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.mm(ge_bar, ge_bar.t()) + self.r1 * torch.eye(o1).cuda()
        SigmaHat22 = (1.0 / (m - 1)) * torch.mm(chem_bar, chem_bar.t()) + self.r2 * torch.eye(o2).cuda()

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        D1, V1 = torch.symeig(SigmaHat11, eigenvectors=True)
        D2, V2 = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, self.eps).nonzero()[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, self.eps).nonzero()[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.mm(torch.mm(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.mm(torch.mm(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.mm(torch.mm(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = torch.sqrt(torch.trace(torch.mm(Tval.t(), Tval)))
        else:
            # just the top outdim_size singular values are used
            U, V = torch.symeig(torch.mm(Tval.t(), Tval), eigenvectors=True)
            U = U[torch.gt(U, self.eps).nonzero()[0]]
            U = U.sort()
            corr = torch.sum(torch.sqrt(U[0:self.k]))
        return -corr


class TripletCCALoss(nn.Module):
    def __init__(self, percent_correct=True, *args, **kwargs):
        super().__init__()
        self.percent_correct = percent_correct
        self.triplet_margin = TripletMarginLoss_WU(percent_correct=percent_correct, *args, **kwargs)
        self.cca_loss = CCALoss()

    def forward(self, anchors, positives, negatives):
        loss_cca = self.cca_loss(anchors, positives)
        res_triplet = self.triplet_margin(anchors, positives, negatives)

        if self.percent_correct:
            loss_trip, perc_correct = res_triplet
        else:
            loss_trip = res_triplet

        loss_total = loss_cca + 2*loss_trip

        if self.percent_correct:
            return loss_total, (perc_correct, loss_cca, loss_trip)
        else:
            return loss_total
