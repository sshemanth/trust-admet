import numpy as np
from sklearn.metrics import pairwise_distances

from trust_admet.data.featurize import dataframe_to_fingerprints


def max_tanimoto_similarity(train_df, query_df, smiles_col="canonical_smiles"):
    train_fp = dataframe_to_fingerprints(train_df, smiles_col=smiles_col)
    query_fp = dataframe_to_fingerprints(query_df, smiles_col=smiles_col)

    distances = pairwise_distances(query_fp, train_fp, metric="jaccard")
    similarities = 1.0 - distances

    return similarities.max(axis=1), similarities.mean(axis=1)


def add_applicability_scores(train_df, query_df):
    query_df = query_df.copy()

    max_sim, mean_sim = max_tanimoto_similarity(train_df, query_df)

    query_df["ad_max_tanimoto"] = max_sim
    query_df["ad_mean_tanimoto"] = mean_sim

    query_df["ad_region"] = np.where(
        query_df["ad_max_tanimoto"] >= 0.5,
        "inside",
        "outside",
    )

    return query_df
