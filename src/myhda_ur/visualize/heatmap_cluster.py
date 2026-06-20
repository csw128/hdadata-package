import numpy as np
import pandas as pd
from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap_cluster(
    data: HDAData = None,
    cluster: pd.Series = None,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    gene_sym: str = None
    ) -> None:
    """Plot per-cluster expression heatmaps (trajectories x timepoints):
        - (in subplots) the individual trajectories in a cluster as rows
        and timepoints as columns

    Args:
        data:
            (subset) data object to analyze
        cluster:
            cluster result stored in HDAData.clusters
        key: "X" / "ntr"
            values to compute on
        kind: "trajectory"
            type of cluster elements (only trajectory supported)
        trajectory_id:
            selected trajectories to analyze
        gene_sym:
            gene to visualize. If None, plots mean across all genes
            
    """
    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    k = len(np.unique(cluster_labels))

    vectors, vec_labels = _get_vectors(
        data=data,
        key=key,
        kind=kind,
        trajectory_id=trajectory_id
    )

    # realign vector index to match cluster order
    vectors = vectors[pd.Index(vec_labels).get_indexer(labels)]

    if kind == "trajectory":
        n_timepoints = len(data.trajectories.columns)
        n_genes = vectors.shape[1] // n_timepoints
        timepoints = list(data.trajectories.columns)

        # unflatten: (n_trajs, n_genes, n_timepoints)
        unflattened = vectors.reshape(len(labels), n_genes, n_timepoints)

        if gene_sym is not None:
            if gene_sym not in data.var.index:
                raise ValueError(f"Gene '{gene_sym}' not found in data.var.index.")
            gene_idx = data.var.index.get_loc(gene_sym)
            plot_vectors = unflattened[:, gene_idx, :]  
            value_label = f"{key}"
        else:
            plot_vectors = unflattened.mean(axis=1)     
            value_label = f"Mean {key} across genes"

    else:
        raise ValueError(f"kind='{kind}' is not supported. Only 'trajectory' is supported.")

    clusters_to_plot = list(range(k))
    cmap = sns.color_palette("Spectral", as_cmap=True)

    fig, axes = plt.subplots(
        1, k,
        figsize=(3.5 * k, 4),
        squeeze=False
    )
    axes = axes[0]

    # compute global vmin/vmax for consistent colorscale across subplots
    vmin = plot_vectors.min()
    vmax = plot_vectors.max()

    for plot_idx, cidx in enumerate(clusters_to_plot):
        ax = axes[plot_idx]
        mask = cluster_labels == cidx
        cluster_vecs = plot_vectors[mask]                        
        cluster_names = [l for l, m in zip(labels, mask) if m]

        df = pd.DataFrame(cluster_vecs, index=cluster_names, columns=timepoints)

        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
            cbar=(plot_idx == k - 1),   # colorbar only on last subplot
            cbar_kws={"label": value_label, "shrink": 0.8}
        )

        ax.set_title(f"Cluster {cidx}", fontsize=10)
        ax.set_xlabel("Timepoint", fontsize=8)
        ax.set_ylabel("Trajectory" if plot_idx == 0 else "", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7, rotation=0)

    gene_label = f" - gene {gene_sym}" if gene_sym else ""
    plt.suptitle(f"Cluster Heatmap{gene_label}", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()