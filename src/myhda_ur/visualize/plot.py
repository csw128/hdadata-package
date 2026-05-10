import pandas as pd
import altair as alt


def plot_heatmap(
    sim: pd.DataFrame,
    type: str = None        
    ) -> alt.Chart:
    """Plot interactive heatmap based on similarity matrix.
        Heatmap types:
        - gene vs gene
        - cell vs cell

        Args:
            sim: Similarity dataframe 
            type: "gene" or "cell"

        Returns:
            Heatmap
            
    """
    row_label = sim.index.name or "index"

    # melt wide format to long format (identifier, variable, value)
    df_long = sim.reset_index().melt(
        id_vars="index",                # identifier variable
        var_name="variable",            # name for unpivoted variable column
        value_name="similarity"         # name for value column
    )

    base = alt.Chart(df_long, title=f'{type} vs {type} similarity')

    heatmap = base.mark_rect().encode(
        x=alt.X(f"{row_label}:N", title=f"{type}", sort=None),  # don't sort alphabetically
        y=alt.Y("variable:N", title=f"{type}", sort=None),
        color=alt.Color(
            "similarity:Q",
            scale=alt.Scale(scheme="redpurple", domain=[0, 1]),
            legend=alt.Legend(title="similarity")
        )
    )

    grid = base.mark_rect(
        fill="transparent",
        stroke="white",
        strokeWidth=0.5
    ).encode(
        x=alt.X(f"{row_label}:N", sort=None),
        y=alt.Y("variable:N", sort=None),
        tooltip=[                                       
            alt.Tooltip(f"{row_label}:N", title="gene x"),
            alt.Tooltip("variable:N", title="gene y"),
            alt.Tooltip("similarity:Q", title="similarity", format=".3f")
        ]
    )
    
    return (heatmap + grid).properties(
        width=alt.Step(50),
        height=alt.Step(50)
    )
    