import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from ..clustering.hierarchical import _get_linkage


def plot_heatmap(
    sim: pd.DataFrame,
    type: str = None,
    cluster: bool = False,
    title: str = None
    ) -> go.Figure:
    """Plot similarity heatmap with or without clustered dendograms using plotly.
    
    Args:
        sim_name: 
            similarity dataframe stored
        type: "gene" / "cell" / "trajectories"
            heatmap type of similarity to be used to plot title
        cluster: True / False
            True  = clustered heatmap with dendograms
            False = regular heatmap without cluster
        title: 
            user-specificed title of the plot
            
    Returns:
        Interactive similarity heatmap
        
    """
    labels = list(sim.index)
    n = len(labels)

    # configurations
    CELL_PX        = 40          # fixed matrix cell size in 40x40 px
    DENDRO_PX      = 120         # fixed dendrogram height in 120 px
    MARGIN_TOP     = 60          
    MARGIN_RIGHT   = 120         
    MARGIN_LEFT    = 20
    MARGIN_BOTTOM  = 20
    CHAR_PX        = 7           # approx pixels per character 
    TICK_OFFSET    = 10          # extra padding beyond label length

    max_label_len  = max(len(l) for l in labels)
    label_px       = max_label_len * CHAR_PX + TICK_OFFSET

    margin_left_y = MARGIN_LEFT + label_px   
    margin_bottom_x = MARGIN_BOTTOM + label_px   

    heatmap_px = n * CELL_PX
    dendro_px  = DENDRO_PX if cluster else 0

    # final figure size
    total_w = margin_left_y + heatmap_px + MARGIN_RIGHT
    total_h = MARGIN_TOP + dendro_px + heatmap_px + margin_bottom_x

    # dendrogram as fraction of total plot height 
    dendro_frac  = dendro_px  / (dendro_px + heatmap_px)
    heatmap_frac = heatmap_px / (dendro_px + heatmap_px)

    if cluster:
        # hierarchical clustering
        Z, _ = _get_linkage(sim)
        dendro_info = dendrogram(Z, labels=labels, no_plot=True)
        order = dendro_info["leaves"]
        ordered_labels = [labels[i] for i in order]
        sim_ordered = sim.iloc[order, :].iloc[:, order]

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[dendro_frac, heatmap_frac],
            vertical_spacing=0.0,
        )

        def dendro_traces(Z, labels):
            info = dendrogram(Z, labels=labels, no_plot=True)
            x_coord_branch = np.array(info["icoord"])
            y_coord_branch = np.array(info["dcoord"])
            traces = []
            for xs, ys in zip(x_coord_branch, y_coord_branch):
                traces.append(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(color="gray", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            return traces

        for tr in dendro_traces(Z, ordered_labels):
            fig.add_trace(tr, row=1, col=1)

        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)

        heatmap_row = 2

    else:
        ordered_labels = labels
        sim_ordered = sim
        fig = go.Figure()
        heatmap_row = None

    heatmap = go.Heatmap(
        z=sim_ordered.values,
        x=ordered_labels,
        y=ordered_labels,
        colorscale="Spectral",
        colorbar=dict(title="similarity", len=0.85, y=0.4),
        hovertemplate=(
            f"<b>{type} x</b>: %{{x}}<br>"
            f"<b>{type} y</b>: %{{y}}<br>"
            f"<b>similarity</b>: %{{z:.3f}}<extra></extra>"
        ),
    )

    if heatmap_row:
        fig.add_trace(heatmap, row=heatmap_row, col=1)
        fig.update_xaxes(tickangle=45, row=heatmap_row, col=1)
        fig.update_yaxes(autorange="reversed", row=heatmap_row, col=1)
    else:
        fig.add_trace(heatmap)
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(autorange="reversed")

    if title is None:
        title = f"{type} vs {type} similarity"

    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=int(total_w),
        height=int(total_h),
        margin=dict(l=int(margin_left_y), r=MARGIN_RIGHT, t=MARGIN_TOP, b=int(margin_bottom_x)),
    )

    return fig
