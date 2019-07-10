import torch
import numpy as np
from torch.utils.data import Dataset
import random


class SiameseL1000Dataset(Dataset):
    def __init__(self, gene_signatures, split = 'train', perts_to_exclude = None,
                 control_sigs = None, rankTransform = False):
        self.includeControls = control_sigs is not None
        self.perts_to_exclude = perts_to_exclude

        # Define cell lines for splits
        self.source_cell_lines = ['A549', 'HT29', 'VCAP', 'HCC515', 'HA1E', 'HEPG2']

        if split == "train":
            self.query_cell_lines = ['A549', 'HT29', 'VCAP', 'HCC515', 'HA1E', 'HEPG2']
        elif split == "val":
            self.query_cell_lines = ['MCF7']
        elif split == "test":
            self.query_cell_lines = ['A375', 'PC3']
        elif split == "neuro":
            self.query_cell_lines = ['NEU', 'FIBRNPC', 'NPC']
        else:
            if isinstance(split, (list,)):
                self.query_cell_lines = split
            else:
                self.query_cell_lines = [split]

        if rankTransform:
            # uniform between 0 and 1.  Can consider other ranges
            gene_signatures = (gene_signatures.rank(axis = 1) - 1 ) / (gene_signatures.shape[1] - 1)
            if control_sigs is not None:
                control_sigs = (control_sigs.rank(axis = 1) - 1 ) / (control_sigs.shape[1] - 1)

        # Subset out the data into source and query datasets
        self.source_sigs = gene_signatures \
            .query('cell_id in @self.source_cell_lines ')

        source_locs = [x in self.source_cell_lines \
                       for x in gene_signatures.index.get_level_values("cell_id").to_list()]
        self.source_sigs = gene_signatures.loc[source_locs]

        query_locs = [x in self.query_cell_lines \
                       for x in gene_signatures.index.get_level_values("cell_id").to_list()]
        self.query_sigs = gene_signatures.loc[query_locs]

        if control_sigs is not None:
            self.source_baseline_sigs = control_sigs.loc[source_locs]
            self.query_baseline_sigs = control_sigs.loc[query_locs]

        if perts_to_exclude is not None:
            self.query_sigs = self.query_sigs \
                .query('pert_id not in @perts_to_exclude')

            keep_locs = [x not in perts_to_exclude \
                           for x in self.query_sigs.index.get_level_values("cell_id").to_list()]

            self.query_sigs = self.query_sigs.loc[keep_locs]
            if control_sigs is not None:
                self.query_baseline_sigs = self.query_baseline_sigs.loc[keep_locs]

        if split != "train":
            # Filter out any drugs from the query dataset that don't have at least one example in training
            source_perts = self.source_sigs \
                .index.get_level_values('pert_id') \
                .unique().to_list()

            locs_queries_in_source = [x in source_perts \
                           for x in self.query_sigs.index.get_level_values("pert_id").to_list()]
            self.query_sigs = self.query_sigs.loc[locs_queries_in_source]


    def __getitem__(self, idx):
        sig_query = self.query_sigs.iloc[idx, :].values
        pid_query = self.query_sigs.index.get_level_values('pert_id')[idx]

        sigs_match = random.randint(0, 1)
        if sigs_match:
            matching_inds = (self.source_sigs
                .index.get_level_values('pert_id') == pid_query)
        else:
            matching_inds = (self.source_sigs
                             .index.get_level_values('pert_id') != pid_query)

        idx_key = np.random.choice(np.where(matching_inds)[0])
        sig_key = self.source_sigs.iloc[idx_key, :].values

        sig_query = torch.tensor(sig_query, dtype=torch.float)
        sig_key = torch.tensor(sig_key, dtype=torch.float)

        if self.includeControls:
            control_query = torch.tensor(self.query_baseline_sigs.iloc[idx, :].values, dtype=torch.float)
            control_key = torch.tensor(self.source_baseline_sigs.iloc[idx_key, :].values, dtype=torch.float)

            sig_query = torch.stack((sig_query, control_query), dim=0)
            sig_key = torch.stack((sig_key, control_key), dim=0)

        label = torch.from_numpy(np.array([sigs_match], dtype=np.float32))

        return (sig_query, sig_key), label

    def __len__(self):
        return len(self.query_sigs)
