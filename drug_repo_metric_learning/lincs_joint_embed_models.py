import torch
import torch.nn as nn
import sys

sys.path.append('/home/sgf2/DBMI_server/repo/chemprop')
from chemprop.models.model import build_model

# Following is the code needed to run a more advanced encoder
#from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
#rd_process = RDKit2DNormalized().process
#feats = [rd_process(x)[1:] for x in smiles_list]
#chemprop_model(smiles_list, feats)

def load_chemprop_model(chemprop_model_path="../chemprop/pcba/model_unoptimized.pt"):
    chemprop_info = torch.load(chemprop_model_path)
    chemprop_model = build_model(chemprop_info['args'])
    chemprop_model.load_state_dict(chemprop_info['state_dict'])
    chemprop_encoder = chemprop_model.encoder
    chemprop_encoder.cuda()
    return chemprop_encoder


class FFANN_Embedder(nn.Module):
    # Written by Matthew, but Sam is adding n_genes as a parameter
    # TODO(mmd): Parametrize, dropout, SNN
    # TODO(mmd): Fix dropout to respect parameter passed in.
    def __init__(
        self, dim_sizes, n_genes = 978, dropout_input=False, dropout_prob=0.5, act=nn.SELU, dropout=nn.AlphaDropout
    ):
        super().__init__()
        assert type(dropout_input) is bool, "Must pass boolean for dropout_input"

        layers = [dropout(dropout_prob)] if dropout_input else []
        in_features = n_genes
        for dim in dim_sizes:
            layers.extend([nn.Linear(in_features, dim), act(), dropout(dropout_prob)])
            in_features = dim

        self.model   = nn.Sequential(*layers)
        self.out_dim = dim_sizes[-1]

    def forward(self, gene_expr): return self.model(gene_expr)


class SNN_Embedder(FFANN_Embedder):
    # Written by Matthew
    def __init__(self, dim_sizes, n_genes = 978, dropout_prob=0.5):
        super().__init__(dim_sizes, n_genes = n_genes, dropout_prob=dropout_prob, act=nn.SELU, dropout=nn.AlphaDropout)

class FeedForwardTripletNet(nn.Module):
    def __init__(self, embed_size=128, input_type="triplet_ge_first",
                 n_genes=978*3, dropout_prob=0.5):
        super().__init__()
        self.input_type = input_type
        self.ge_embed = SNN_Embedder(dim_sizes=[1024, 512, embed_size], n_genes=n_genes, dropout_prob=dropout_prob)

        self.chemprop_encoder = load_chemprop_model()
        self.chem_linear = SNN_Embedder(dim_sizes=[embed_size], n_genes=300, dropout_prob=dropout_prob)

        self.ge_embed.cuda(); self.chem_linear.cuda();

    def forward(self, input):
        if self.input_type == "triplet_ge_first":
            anchor = self.ge_embed(input[0].cuda())
            match = self.chem_linear(self.chemprop_encoder(list(input[1])))
            non_match = self.chem_linear(self.chemprop_encoder(list(input[2])))
        elif self.input_type == "triplet_ge_only":
            anchor = self.ge_embed(input[0].cuda())
            match = self.ge_embed(input[1].cuda())
            non_match = self.ge_embed(input[2].cuda())
        elif self.input_type == "triplet_chem_first":
            anchor = self.chem_linear(self.chemprop_encoder(list(input[0])))
            match = self.ge_embed(input[1].cuda())
            non_match = self.ge_embed(input[2])
        return anchor, match, non_match


class FeedForwardQuadrupletNet(FeedForwardTripletNet):
    def __init__(self, *args, **kwargs):
        super().__init__(input_type="quadruplet", *args, **kwargs)

    def forward(self, input):
        anchor_ge, non_match_ge, anchor_chem, non_match_chem = input

        anchor_ge = self.ge_embed(anchor_ge.cuda())
        non_match_ge = self.ge_embed(non_match_ge.cuda())
        anchor_chem = self.chem_linear(self.chemprop_encoder(list(anchor_chem)))
        non_match_chem = self.chem_linear(self.chemprop_encoder(list(non_match_chem)))

        return anchor_ge, non_match_ge, anchor_chem, non_match_chem

class FeedForwardQuintupletNet(FeedForwardTripletNet):
    class LincsQuintupletDataset(FeedForwardTripletNet):
    def __init__(self, *args, **kwargs):
        super().__init__(input_type="quintuplet", *args, **kwargs)

    def forward(self, input):
        anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem = input

        anchor_ge = self.ge_embed(anchor_ge.cuda())
        match_ge = self.ge_embed(match_ge.cuda())
        non_match_ge = self.ge_embed(non_match_ge.cuda())
        anchor_chem = self.chem_linear(self.chemprop_encoder(list(anchor_chem)))
        non_match_chem = self.chem_linear(self.chemprop_encoder(list(non_match_chem)))

        return anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem
