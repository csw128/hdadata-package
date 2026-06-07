from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_line(
    data: HDAData = None,
    cluster: pd.Series = None,            
    key: str = None,
    kind: str = None,
    trajectory_id: list = None
    ) -> None:

    labels = cluster.index.tolist()
    cluster_labels = cluster.values
    
    k = len(np.unique(cluster_labels))
    
    vectors, _ = _get_vectors(
                            data=data,
                            key=key,
                            kind=kind,
                            trajectory_id=trajectory_id
                        )
    
    if kind == "trajectory":
        n_timepoints = len(data.trajectories.columns)
        n_genes = vectors.shape[1] // n_timepoints

        # unflatten trajectory vector to get mean across gene
        plot_vectors = vectors.reshape(len(labels), n_genes, n_timepoints).mean(axis=1)
        xlabel = "Timepoint"
        ylabel = f'Mean {key} across genes'
        timepoints = data.trajectories.columns
    
    elif kind == "gene":
        plot_vectors = vectors
        xlabel = "Cell index"
        ylabel = f'{key} value'
        timepoints = None

    elif kind == "cell":
        plot_vectors = vectors
        xlabel = "Gene index"
        ylabel = f'{key} value'
        timepoints = None

    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4), sharey=True)
    cmap = plt.get_cmap("tab10")
    
    cluster_means = []

    for cluster_idx in range(k):
        ax = axes[cluster_idx]
        mask = cluster_labels == cluster_idx
        cluster_vecs = plot_vectors[mask]
        cluster_names = [l for l, m in zip(labels, mask) if m]

        # plot individual lines in each cluster
        for vec, name in zip(cluster_vecs, cluster_names):
            ax.plot(vec, alpha=0.4, color=cmap(cluster_idx), linewidth=1)

        # plot cluster mean line
        mean_vec = cluster_vecs.mean(axis=0)
        cluster_means.append(mean_vec)
        ax.plot(mean_vec, color=cmap(cluster_idx), linewidth=2.5, linestyle="--", label="mean")

        if kind == "trajectory" and timepoints is not None:
            ax.set_xticks(range(len(timepoints)))
            ax.set_xticklabels(timepoints)
            
        ax.set_title(f"Cluster {cluster_idx}")
        ax.set_xlabel(xlabel)
        if cluster_idx == 0:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper left")

    # plot combined overlap of cluster mean lines
    ax_combined = axes[-1]
    for cluster_idx, mean_vec in enumerate(cluster_means):
        ax_combined.plot(mean_vec, color=cmap(cluster_idx), linewidth=2.5, label=f"Cluster {cluster_idx}")
    
    if kind == "trajectory" and timepoints is not None:
        ax_combined.set_xticks(range(len(timepoints)))
        ax_combined.set_xticklabels(timepoints)
    
    ax_combined.set_title("Cluster means")
    ax_combined.set_xlabel(xlabel)
    ax_combined.legend(fontsize=7, loc="upper left")

    plt.suptitle(f'Overview of Cluster Analysis', y=1.02)
    plt.tight_layout()
    plt.show()
