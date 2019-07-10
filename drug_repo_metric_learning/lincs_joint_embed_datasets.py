import torch
import numpy as np
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

def scale(df, m, s): return (df - m)/s


class LincsTripletDataset(Dataset):
    def __init__(self, l1000_sigs, control_method = 'baseline_logFC',
                 split = 'train', train_test_seed = 17,
                 pert_time="24.0", pert_dose = "um@10.0", cell_id = "PC3",
                 rank_transform=False, perts_to_exclude=None, input_type="triplet_ge_first"
                ):
        self.control_method = control_method
        self.perts_to_exclude = perts_to_exclude
        self.pert_time = pert_time
        self.pert_dose = pert_dose
        self.cell_id = cell_id if isinstance(cell_id, list) else [cell_id]
        self.input_type = input_type

        # Define cell lines for splits
        nan = np.nan
        l1000_perts = l1000_sigs.query('pert_time == @pert_time ' +
                                       '& pert_dose == @pert_dose ' +
                                       '& cell_id in @self.cell_id ' +
                                       '& canonical_smiles not in ["-666", "restricted", @nan] ')
        l1000_perts_plates = list(set(l1000_perts.index.get_level_values('plate').values))

        l1000_controls = l1000_sigs.query('pert_time == @pert_time' +
                                          '& cell_id in @self.cell_id ' +
                                          '& pert_type in ["ctl_vehicle", "ctl_vector", "ctl_untrt"] ' +
                                          '& plate in @l1000_perts_plates')

        # Train-test split by plate
        plates_train, plates_non_train = train_test_split(l1000_perts_plates, test_size=0.2,
                                                          random_state=train_test_seed)
        plates_val, plates_test = train_test_split(plates_non_train, test_size=0.5, random_state=train_test_seed)

        if split == 'train':
            l1000_perts = l1000_perts.query('plate in @plates_train')
            l1000_controls = l1000_controls.query('plate in @plates_train')
        elif split == 'val':
            l1000_perts = l1000_perts.query('plate in @plates_val')
            l1000_controls = l1000_controls.query('plate in @plates_val')
        else:
            l1000_perts = l1000_perts.query('plate in @plates_test')
            l1000_controls = l1000_controls.query('plate in @plates_test')

        # Save raw values for fold-change
        self.l1000_perts_raw = l1000_perts
        self.l1000_controls_raw = l1000_controls

        # Standardize measurements
        l1000_on_train_plates = l1000_sigs.query('pert_time == @pert_time ' +
                                                 '& cell_id in @self.cell_id ' +
                                                 '& plate in @plates_train')
        mu, sigma = l1000_on_train_plates.mean(axis=0), l1000_on_train_plates.std(axis=0)
        l1000_perts, l1000_controls = [scale(d, mu, sigma) for d in (l1000_perts, l1000_controls)]

        # Rank transformation -- currently scales ranks uniformly between 0 and 1; consider adjusting
        if rank_transform:
            l1000_perts = (l1000_perts.rank(axis=1) - 1) / (l1000_perts.shape[1] - 1)
            l1000_controls = (l1000_controls.rank(axis=1) - 1) / (l1000_controls.shape[1] - 1)

        # Store results
        self.l1000_perts = l1000_perts
        self.l1000_controls = l1000_controls

        # Create distance matrix,
        pert_mean_sigs = self.l1000_perts.groupby('pert_id').mean()
        self.pert_names = pert_mean_sigs.index.values
        self.pert_dist = cdist(pert_mean_sigs, pert_mean_sigs, metric='correlation')
        self.pert_dist_min = np.min(self.pert_dist)
        self.pert_dist_max = np.max(self.pert_dist)

    def __get_GE_sig__(self, idx):
        # Post-pert signature
        pert_sig = self.l1000_perts.iloc[idx, :].values
        pert_sig = torch.tensor(pert_sig, dtype=torch.float)
        pert_sig_raw = self.l1000_perts_raw.iloc[idx, :].values
        pert_sig_raw = torch.tensor(pert_sig_raw, dtype=torch.float)

        # Pre-pert signature
        if self.control_method is not None:
            pert_plate = self.l1000_perts.index.get_level_values('plate')[idx]
            indices_on_plate = set(np.where(self.l1000_controls.index.get_level_values("plate").values == pert_plate)[0])
            idx_ctrl = random.sample(indices_on_plate, 1)[0]
            ctrl_sig = self.l1000_controls.iloc[idx_ctrl, :].values
            ctrl_sig = torch.tensor(ctrl_sig, dtype=torch.float)
            if self.control_method in ["all", "baseline_logFC"]:
                ctrl_sig_raw = self.l1000_controls_raw.iloc[idx_ctrl, :].values
                ctrl_sig_raw = torch.tensor(ctrl_sig_raw, dtype=torch.float)

        # Stack signatures together
        epsilon = 1e-6
        if self.control_method == "baseline":
            ge_sigs = torch.cat((ctrl_sig, pert_sig),
                                dim=0)
        elif self.control_method == "baseline_logFC":
            ge_sigs = torch.cat((ctrl_sig, pert_sig,
                                 torch.log2( (pert_sig_raw + epsilon) / (ctrl_sig_raw + epsilon))),
                                dim=0)
        elif self.control_method == "all":
            ge_sigs = torch.cat((ctrl_sig, pert_sig,
                                 (pert_sig - ctrl_sig),
                                 torch.log2( (pert_sig_raw + epsilon) / (ctrl_sig_raw + epsilon))),
                                dim=0)
            # could convert above to channels with:
            # torch.stack(torch.chunk(ge_sigs, ge_sigs.shape[0]/978), 0)
        else:
            ge_sigs = pert_sig
        return ge_sigs

    def __get_ge_match_idx__(self, idx):
        # Sample a GE signature from a different experiment with the same drug
        pert_id = self.l1000_perts.index.get_level_values("pert_id")[idx]
        indices_matches = set(np.where(self.l1000_perts.index.get_level_values("pert_id").values == pert_id)[0])
        if len(indices_matches) > 1:
            indices_matches = indices_matches - {idx}
        return random.sample(indices_matches, 1)[0]

    def __get_second_pert_idx__(self, idx):
        # DOES NOT CURRENTLY INCORPORATE DISTANCE CLIPPING
        # Uniformly samples in average GE distance from current pert
        # Then randomly picks an example from that perturbagen
        pert_name_to_avoid = self.l1000_perts.index.get_level_values("pert_id").values[idx]
        sample_pert_name = pert_name_to_avoid
        while sample_pert_name == pert_name_to_avoid:
            idx_sample_pert = np.where(self.pert_names == sample_pert_name)[0][0]
            target_dist = np.random.uniform(low=self.pert_dist_min, high=self.pert_dist_max)
            new_sample_pert_idx = (np.abs(self.pert_dist[idx_sample_pert, :] - target_dist)).argmin()
            sample_pert_name = self.pert_names[new_sample_pert_idx]
        indices_matches = set(np.where(self.l1000_perts.index.get_level_values("pert_id").values == sample_pert_name)[0])
        idx_pair = random.sample(indices_matches, 1)[0]
        return idx_pair

    def __get_smiles__(self, idx):
        return self.l1000_perts.index.get_level_values('canonical_smiles').values[idx]

    def __getitem__(self, idx):
        idx_non_match = self.__get_second_pert_idx__(idx)

        if self.input_type == "triplet_ge_first":
            anchor = self.__get_GE_sig__(idx)
            match = self.__get_smiles__(idx)
            non_match = self.__get_smiles__(idx_non_match)
        elif self.input_type == "triplet_ge_only":
            anchor = self.__get_GE_sig__(idx)
            match = self.__get_GE_sig__(self.__get_ge_match_idx__(idx))
            non_match = self.__get_GE_sig__(idx_non_match)
        elif self.input_type == "triplet_chem_first":
            anchor = self.__get_smiles__(idx)
            match = self.__get_GE_sig__(idx)
            non_match = self.__get_GE_sig__(idx_non_match)
        else:
            raise NotImplementedError
        return anchor, match, non_match

    def __len__(self):
        return self.l1000_perts.shape[0]


# Contrastive training
class LincsContrastiveDataset(LincsTripletDataset):
    def __init__(self, l1000_sigs, input_type="contrastive_multimodal", *args, **kwargs):
        super().__init__(l1000_sigs, input_type=input_type, *args, **kwargs)

    def __getitem__(self, idx):
        ge_sigs = self.__get_GE_sig__(idx)

        # Decide if match or non-match
        sigs_match = random.randint(0, 1)
        label = torch.from_numpy(np.array([sigs_match], dtype=np.float32))
        if sigs_match:
            idx_pair = idx
        else:
            idx_pair = self.__get_second_pert_idx__(idx)

        # Create output
        if self.input_type == "contrastive_ge_only":
            pair = self.__get_GE_sig__(self.__get_ge_match_idx__(idx_pair))
        else: # input_type == "contrastive_multimodal"
            pair = self.__get_smiles__(idx_pair)

        return (ge_sigs, pair), label


# Idea here is to do margin loss with anchor_ge vs anchor_chem/non_match_chem
# and additionally do the margin loss with anchor_chem vs anchor_ge/non_match_ge
# Each one gets an output and you backprop over both
# Could temporarily call this a double-margin loss
class LincsQuadrupletDataset(LincsTripletDataset):
    def __init__(self, l1000_sigs, *args, **kwargs):
        super().__init__(l1000_sigs, input_type="quadruplet", *args, **kwargs)

    def __getitem__(self, idx):
        idx_non_match = self.__get_second_pert_idx__(idx)

        anchor_ge = self.__get_GE_sig__(idx)
        non_match_ge = self.__get_GE_sig__(idx_non_match)
        anchor_chem = self.__get_smiles__(idx)
        non_match_chem = self.__get_smiles__(idx_non_match)

        return anchor_ge, non_match_ge, anchor_chem, non_match_chem


# Idea here is to do
# margin loss with anchor_ge vs anchor_chem/non_match_chem
# margin loss with anchor_chem vs anchor_ge/non_match_ge
# margin loss with anchor_ge vs match_ge/non_match_ge
class LincsQuintupletDataset(LincsTripletDataset):
    def __init__(self, l1000_sigs, *args, **kwargs):
        super().__init__(l1000_sigs, input_type="quintuplet", *args, **kwargs)

    def __getitem__(self, idx):
        idx_chem_non_match = self.__get_second_pert_idx__(idx)
        idx_ge_match = self.__get_ge_match_idx__(idx)

        anchor_ge = self.__get_GE_sig__(idx)
        match_ge = self.__get_GE_sig__(idx_ge_match)
        non_match_ge = self.__get_GE_sig__(idx_chem_non_match)
        anchor_chem = self.__get_smiles__(idx)
        non_match_chem = self.__get_smiles__(idx_chem_non_match)

        return anchor_ge, match_ge, non_match_ge, anchor_chem, non_match_chem


