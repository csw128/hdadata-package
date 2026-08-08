import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


# def plot_scatter(
#     cluster: pd.Series = None,           
#     embedding: tuple = None,
#     title: str = None,
#     show_label: bool = False
#     ) -> None:
#     """Plot scatter to visualize the clusters formed from the chosen clustering
#        algorithm in a reduced dimension.
    
#     Args:
#         cluster:
#             cluster result stored in HDAData.clusters
#         embedding:
#             embedding result stored in HDAData.embeddings
#         title:
#             title of the plot
#         show_label: True / False
#             display the gene / cell / traj label next to the data point
            
#     """
#     labels = cluster.index.tolist()
#     cluster_labels = cluster.values
    
#     coords, xlabel, ylabel = embedding
    
#     if title is None:
#         title = f'Clustering on {cluster.name}'
        
#     cmap = plt.get_cmap("Paired")
#     norm = mpl.colors.Normalize(vmin=0, vmax=9)          
#     unique_labels = sorted(set(cluster_labels))
#     color_map = {label: cmap(norm(label)) for label in unique_labels}
      
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     for label in unique_labels:
#         mask = cluster_labels == label
#         ax.scatter(
#             coords[mask, 0], coords[mask, 1],
#             color=color_map[label],
#             label=str(label)
#         )

#     if show_label:
#         for i, name in enumerate(labels):
#             ax.annotate(name, coords[i], fontsize=7, ha="left")
        
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
    
#     ax.legend(
#         title="Cluster",
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left",
#         borderaxespad=0,
#         frameon=True
#     )
    
#     plt.tight_layout()
#     plt.show()


def plot_scatter(
    embedding: tuple = None,
    cluster: pd.Series = None,
    highlight: object = None,
    values: pd.Series = None,
    mask: pd.Series = None,
    title: str = None,
    show_label: bool = False,
    legend_label: str = None,
    clip_percentile: tuple = (5, 95)
    ) -> None:
    """Flexible scatter plot on a fixed UMAP embedding. Supports four modes,
       chosen by which arguments are given:

        1. cluster only                -> all clusters, each its own color
                                           (equivalent to the old plot_scatter)
        2. cluster + highlight         -> only that cluster colored, rest grey
        3. values only                 -> continuous color across ALL cells
        4. mask only                   -> discrete highlight: masked cells
                                           colored, rest grey (e.g. timepoint)
        5. values + mask               -> continuous color restricted to
                                           masked cells, rest grey

    """
    if cluster is None and values is None and mask is None:
        raise ValueError("Provide at least one of `cluster`, `values`, or `mask`.")

    coords, xlabel, ylabel = embedding
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = "RdYlBu"

    # Mode 1: cluster given, no highlight -> all clusters, own colors
    if cluster is not None and highlight is None:
        cell_names = cluster.index.tolist()
        cluster_labels = cluster.values

        cmap_cluster = plt.get_cmap("Paired")
        norm = mpl.colors.Normalize(vmin=0, vmax=9)
        unique_labels = sorted(set(cluster_labels))
        color_map = {label: cmap_cluster(norm(label)) for label in unique_labels}

        for label in unique_labels:
            m = cluster_labels == label
            ax.scatter(
                coords[m, 0], coords[m, 1],
                color=color_map[label],
                label=str(label)
            )

        ax.legend(
            title="Cluster",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=True
        )

    else:
        highlight_color = None

        if cluster is not None:
            # Mode 2: cluster + highlight -> convert to mask + fixed color
            unique_labels = sorted(set(cluster.values))
            if highlight not in unique_labels:
                raise ValueError(
                    f"highlight={highlight!r} not found in cluster labels {unique_labels}"
                )

            cmap_cluster = plt.get_cmap("Paired")
            norm = mpl.colors.Normalize(vmin=0, vmax=9)
            highlight_color = cmap_cluster(norm(highlight))

            cluster_mask = cluster == highlight
            mask = cluster_mask if mask is None else (mask.reindex(cluster.index) & cluster_mask)

            if legend_label is None:
                legend_label = f"cluster {highlight}"

        ref = values if values is not None else (mask if mask is not None else cluster)
        cell_names = ref.index.tolist()

        if values is not None and mask is not None:
            mask = mask.reindex(values.index)

        # compute clip range from the full values (mode 3 and mode 5 always share the same color scale regardless of masking)
        if values is not None:
            vmin, vmax = np.percentile(values.values, clip_percentile)
        
        if values is not None and mask is None:
            # Mode 3: continuous color across all cells
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values.values, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(legend_label or values.name or "value")

        elif mask is not None and values is None:
            # Mode 2/4: discrete highlight
            mask_vals = mask.values.astype(bool)
            color = highlight_color if highlight_color is not None else "tab:red"

            ax.scatter(coords[~mask_vals, 0], coords[~mask_vals, 1], color="lightgrey", label="_nolegend_", zorder=1)
            ax.scatter(coords[mask_vals, 0], coords[mask_vals, 1], color=color, label=legend_label or "highlighted", zorder=2)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)

        else:
            # Mode 5: continuous restricted to masked cells
            mask_vals = mask.values.astype(bool)

            ax.scatter(coords[~mask_vals, 0], coords[~mask_vals, 1], color="lightgrey", label="_nolegend_", zorder=1)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(legend_label or values.name or "value")

    if show_label:
        for i, name in enumerate(cell_names):
            ax.annotate(name, coords[i], fontsize=7, ha="left")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()