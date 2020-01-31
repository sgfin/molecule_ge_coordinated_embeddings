import torch
import torch.nn as nn
import sys

sys.path.append('/home/sgf2/DBMI_server/repo/chemprop')
from chemprop.models.model import build_model

def load_chemprop_model(chemprop_model_path):
    chemprop_info = torch.load(chemprop_model_path)
    chemprop_model = build_model(chemprop_info['args'])
    chemprop_model.load_state_dict(chemprop_info['state_dict'])
    chemprop_encoder = chemprop_model.encoder
    chemprop_encoder.cuda()
    return chemprop_encoder


class FFANN_Embedder(nn.Module):
    # Written by Matthew, but Sam is adding n_feats as a parameter
    def __init__(
        self, dim_sizes, n_feats=978, linear_bias=True,
            dropout_prob=0, act=nn.ReLU, dropout=nn.Dropout,
            dropout_input=False
    ):
        super().__init__()
        assert type(dropout_input) is bool, "Must pass boolean for dropout_input"

        layers = [dropout(dropout_prob)] if dropout_input else []
        in_features = n_feats
        for i,dim in enumerate(dim_sizes):
            layers.extend([nn.Linear(in_features, dim, bias=linear_bias)])
            if i < len(dim_sizes)-1:
                layers.extend([act(), dropout(dropout_prob)])
            in_features = dim

        self.model = nn.Sequential(*layers)
        self.out_dim = dim_sizes[-1]

    def forward(self, gene_expr): return self.model(gene_expr)


class SNN_Embedder(FFANN_Embedder):
    # Written by Matthew McDermott
    def __init__(self, dim_sizes, n_feats=978, dropout_prob=0, act=nn.SELU, linear_bias=True):
        super().__init__(dim_sizes, n_feats=n_feats,
                         dropout_prob=dropout_prob, act=act, dropout=nn.AlphaDropout,
                         linear_bias=linear_bias)


class FeedForwardGExChemNet(nn.Module):
    def __init__(self, embed_size=128, n_feats_genes=978,
                 hidden_layers_ge=[1024, 512], hidden_layers_chem=[],
                 dropout_prob=0, act="selu", linear_bias=True,
                 input_type="singlet",
                 chemprop_model_path="/home/sgf2/DBMI_server/repo/chemprop/pcba/model_unoptimized.pt",
                 pretrained_model_path=None,
                 smiles_to_feats=None):
        super().__init__()
        self.input_type = input_type
        self.embed_size = embed_size
        self.smiles_to_feats = smiles_to_feats
        self.rdkit = smiles_to_feats is not None

        assert act in ("sigmoid", "relu", "tanh", "selu"), "Unsupported activation: %s!" % act

        Embedder = FFANN_Embedder
        if act == "sigmoid": act = nn.Sigmoid
        elif act == "relu": act = nn.ReLU
        elif act == "tanh": act = nn.Tanh
        else:
            act = nn.SELU
            Embedder = SNN_Embedder

        if pretrained_model_path is None:
            # GE Embedder
            ge_layers = hidden_layers_ge + [embed_size]
            self.ge_embed = Embedder(
                dim_sizes=ge_layers, n_feats=n_feats_genes, dropout_prob=dropout_prob, act=act,
                linear_bias=linear_bias
            )

            # Chemprop Embedder
            self.chemprop_encoder = load_chemprop_model(chemprop_model_path)
            chem_layers = hidden_layers_chem + [embed_size]
            if self.rdkit:
                n_feats_chemprop = 2400
            else:
                n_feats_chemprop = 300
            self.chem_linear = Embedder(
                dim_sizes=chem_layers, n_feats=n_feats_chemprop, dropout_prob=dropout_prob, act=act,
                linear_bias=linear_bias
            )
        else:
            model = torch.load(pretrained_model_path)
            self.ge_embed = model.ge_embed
            self.chemprop_encoder = model.chemprop_encoder
            self.chem_linear = model.chem_linear

        # Move to GPU
        self.ge_embed.cuda()
        self.chemprop_encoder.cuda()
        self.chem_linear.cuda()

    def forward(self, input):
        ge, chem = input
        smiles = list(chem)
        if self.rdkit:
            feats = [self.smiles_to_feats[x] for x in smiles]
            chem_encod = self.chemprop_encoder(smiles, feats)
        else:
            chem_encod = self.chemprop_encoder(smiles)
        ge = self.ge_embed(ge.cuda())
        chem = self.chem_linear(chem_encod)
        return ge, chem


class FeedForwardTripletNet(FeedForwardGExChemNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.input_type == "triplet_chem_first":
            smiles = list(input[0])
            if self.rdkit:
                feats = [self.smiles_to_feats[x] for x in smiles]
                chem_encod = self.chemprop_encoder(smiles, feats)
            else:
                chem_encod = self.chemprop_encoder(smiles)
            anchor = self.chem_linear(chem_encod)
            match = self.ge_embed(input[1].cuda())
            non_match = self.ge_embed(input[2].cuda())
        elif self.input_type == "triplet_ge_first":
            smiles_1 = list(input[1])
            smiles_2 = list(input[2])
            if self.rdkit:
                feats_1 = [self.smiles_to_feats[x] for x in smiles_1]
                feats_2 = [self.smiles_to_feats[x] for x in smiles_2]
                chem_encod_1 = self.chemprop_encoder(smiles_1, feats_1)
                chem_encod_2 = self.chemprop_encoder(smiles_2, feats_2)
            else:
                chem_encod_1 = self.chemprop_encoder(smiles_1)
                chem_encod_2 = self.chemprop_encoder(smiles_2)
            anchor = self.ge_embed(input[0].cuda())
            match = self.chem_linear(chem_encod_1)
            non_match = self.chem_linear(chem_encod_2)
        elif self.input_type == "triplet_chem_first":
            smiles = list(input[0])
            if self.rdkit:
                feats = [self.smiles_to_feats[x] for x in smiles]
                chem_encod = self.chemprop_encoder(smiles, feats)
            else:
                chem_encod = self.chemprop_encoder(smiles)
            anchor = self.chem_linear(chem_encod)
            match = self.ge_embed(input[1].cuda())
            non_match = self.ge_embed(input[2].cuda())
        elif self.input_type == "triplet_ge_only":
            anchor = self.ge_embed(input[0].cuda())
            match = self.ge_embed(input[1].cuda())
            non_match = self.ge_embed(input[2].cuda())
        return anchor, match, non_match


class FeedForwardQuadrupletNet(FeedForwardGExChemNet):
    def __init__(self, *args, **kwargs):
        kwargs['input_type'] = 'quadruplet'
        super().__init__(*args, **kwargs)

    def forward(self, input):
        anchor_ge, non_match_ge, anchor_chem, non_match_chem = input
        smiles_anchor = list(anchor_chem)
        smiles_nonmatch = list(non_match_chem)
        if self.rdkit:
            feats_anchor = [self.smiles_to_feats[x] for x in smiles_anchor]
            feats_nonmatch = [self.smiles_to_feats[x] for x in smiles_nonmatch]
            chem_encod_anchor = self.chemprop_encoder(smiles_anchor, feats_anchor)
            chem_encod_nonmatch = self.chemprop_encoder(smiles_nonmatch, feats_nonmatch)
        else:
            chem_encod_anchor = self.chemprop_encoder(smiles_anchor)
            chem_encod_nonmatch = self.chemprop_encoder(smiles_nonmatch)
        anchor_ge = self.ge_embed(anchor_ge.cuda())
        non_match_ge = self.ge_embed(non_match_ge.cuda())
        anchor_chem = self.chem_linear(chem_encod_anchor)
        non_match_chem = self.chem_linear(chem_encod_nonmatch)
        return anchor_ge, non_match_ge, anchor_chem, non_match_chem


class FeedForwardQuintupletNet(FeedForwardGExChemNet):
    def __init__(self, *args, **kwargs):
        kwargs['input_type'] = 'quintuplet'
        super().__init__(*args, **kwargs)

    def forward(self, input):
        anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem = input
        smiles_anchor = list(anchor_chem)
        smiles_nonmatch = list(non_match_chem)
        if self.rdkit:
            feats_anchor = [self.smiles_to_feats[x] for x in smiles_anchor]
            feats_nonmatch = [self.smiles_to_feats[x] for x in smiles_nonmatch]
            chem_encod_anchor = self.chemprop_encoder(smiles_anchor, feats_anchor)
            chem_encod_nonmatch = self.chemprop_encoder(smiles_nonmatch, feats_nonmatch)
        else:
            chem_encod_anchor = self.chemprop_encoder(smiles_anchor)
            chem_encod_nonmatch = self.chemprop_encoder(smiles_nonmatch)
        anchor_ge = self.ge_embed(anchor_ge.cuda())
        match_ge = self.ge_embed(match_ge.cuda())
        non_match_ge = self.ge_embed(non_match_ge.cuda())
        anchor_chem = self.chem_linear(chem_encod_anchor)
        non_match_chem = self.chem_linear(chem_encod_nonmatch)
        return anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem
