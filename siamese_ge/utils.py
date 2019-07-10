from sklearn.model_selection import train_test_split


def generate_pertids_to_exclude(gene_signatures, group = "all", subsample_rate = 1.):
    test_perts = set(gene_signatures.index.get_level_values("pert_id").to_list())

    if group == "all":
        cell_list = ['A549', 'HT29', 'VCAP', 'HCC515', 'HA1E', 'HEPG2', 'MCF7', 'A375', 'PC3', 'NEU', 'FIBRNPC', 'NPC']
    elif group == "non_neuro":
        cell_list = ['A549', 'HT29', 'VCAP', 'HCC515', 'HA1E', 'HEPG2', 'MCF7', 'A375', 'PC3']

    for cid in cell_list:
        cid_perts = set(gene_signatures.query('cell_id == @cid').index.get_level_values("pert_id").to_list())
        test_perts = test_perts.intersection(cid_perts)

    test_perts = list(test_perts)
    if subsample_rate < 1:
        perts_split = train_test_split(test_perts, train_size=subsample_rate, random_state=42)
        test_perts = perts_split[0]

    return test_perts