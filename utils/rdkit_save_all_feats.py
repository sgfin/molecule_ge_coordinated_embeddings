import pandas as pd
import numpy as np
from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
import pickle

nan = np.nan

l1000_sigs_path="~/DBMI_server/repo/mdeg_collab/data/lincs_level3_all_perts.pkl"
l1000_sigs=pd.read_pickle(l1000_sigs_path)
pert_smiles = l1000_sigs.query('pert_type == "trt_cp" & canonical_smiles not in ["-666", "restricted", @nan]').index.get_level_values('canonical_smiles').values

rd_process = {x:RDKit2DNormalized().process(x)[1:] for x in list(set(pert_smiles))}

pickle.dump(rd_process, open( "precomputed/smile_to_rdkit_all.pkl", "wb" ))