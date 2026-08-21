import numpy as np
import pandas as pd
from ..structure.object import HDAData
from ..dynamic_models.representation import _get_vectors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_heatmap_per_cluster(
    data: HDAData = None,
    cluster: pd.Series = None,
    key: str = None,
    kind: str = None,
    trajectory_id: list = None,
    ) -> go.Figure:
    """Plot per-cluster expression heatmaps (genes x timepoints), where each subplot shows mean expression of genes across trajectories in a cluster.
    
        Args:
            data:
                (subset) data object to analyze
            cluster:
                cluster result stored in HDAData.clusters
            key: "X" / "ntr" / "new_rna"
                values to compute on
            kind: "trajectory"
                type of cluster elements (only trajectory supported)
            trajectory_id:
                selected trajectories to analyze

        Returns:
            Per-cluster heatmap 
    
    """
    if kind != "trajectory":
        raise ValueError(f"kind='{kind}' is not supported. Only 'trajectory' is supported.")

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

    n_timepoints = len(data.trajectories.columns)
    n_genes = vectors.shape[1] // n_timepoints
    timepoints = list(data.trajectories.columns)
    gene_names = data.var.index.tolist()

    # unflatten traj vector: (n_trajs, n_genes, n_timepoints)
    unflattened = vectors.reshape(len(labels), n_genes, n_timepoints)

    # per-cluster mean and global vmin/vmax
    cluster_means = []
    for cidx in range(k):
        mask = cluster_labels == cidx
        mean_expr = unflattened[mask].mean(axis=0)  # (n_genes, n_timepoints)
        cluster_means.append(mean_expr)

    vmin = min(m.min() for m in cluster_means)
    vmax = max(m.max() for m in cluster_means)

    # layout
    CELL_PX_X    = 40
    CELL_PX_Y    = 20
    SUBPLOT_W    = CELL_PX_X * n_timepoints
    SUBPLOT_H    = CELL_PX_Y * n_genes
    MARGIN_TOP   = 80
    MARGIN_BOT   = 60
    MARGIN_LEFT  = 80
    MARGIN_RIGHT = 120
    SPACING      = 0.04

    total_w = MARGIN_LEFT + k * SUBPLOT_W + MARGIN_RIGHT + int((k - 1) * SPACING * SUBPLOT_W * k)
    total_h = MARGIN_TOP + SUBPLOT_H + MARGIN_BOT

    fig = make_subplots(
        rows=1, cols=k,
        subplot_titles=[f"Cluster {cidx}" for cidx in range(k)],
        horizontal_spacing=SPACING,
    )

    for cidx in range(k):
        mean_expr = cluster_means[cidx] 

        heatmap = go.Heatmap(
            z=mean_expr,
            x=timepoints,
            y=gene_names,
            colorscale="Spectral",
            zmin=vmin,
            zmax=vmax,
            showscale=(cidx == k - 1),
            colorbar=dict(
                title=f"Mean {key}",
                len=0.85,
                y=0.45,
            ) if cidx == k - 1 else None,
            hovertemplate=(
                f"<b>Gene</b>: %{{y}}<br>"
                f"<b>Timepoint</b>: %{{x}}<br>"
                f"<b>Mean {key}</b>: %{{z:.4f}}<extra></extra>"
            ),
        )

        fig.add_trace(heatmap, row=1, col=cidx + 1)

        fig.update_xaxes(tickangle=45, title_text="Timepoint", row=1, col=cidx + 1)
        fig.update_yaxes(
            showticklabels=False,
            title_text="Gene" if cidx == 0 else "",
            row=1, col=cidx + 1
        )

    fig.update_layout(
        title=f"Cluster Heatmap",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=int(total_w),
        height=int(total_h),
        margin=dict(l=MARGIN_LEFT, r=MARGIN_RIGHT, t=MARGIN_TOP, b=MARGIN_BOT),
    )

    return fig