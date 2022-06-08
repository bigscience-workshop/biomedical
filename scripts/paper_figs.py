import plotly.express as px

import pandas as pd

from parse_single_dataset import IBM_COLORS
from plotly.subplots import make_subplots
import plotly.graph_objects as go

BLURB = ("biosses", "bioasq", "scitail", "mednli", "gad")

df = pd.read_csv("scripts/bigbio_zs_scores_per_prompt.tsv", sep="\t", header=0)

# Initialize figure with subplots
fig = make_subplots(rows=5, cols=1, subplot_titles=(), 
    # vertical_spacing=0.5
    )
for i, dataset in enumerate(BLURB):
    col = i % 3 + 1
    row = i // 3 + 1
    sub_df = df[df["dataset"] == dataset]
    data = []
    for j, p in enumerate(set(sub_df['prompt'])):
        prompt_df = sub_df[sub_df['prompt'] == p]
        prompts_x = [' '.join(mn.split('-')[-2:]) for mn in prompt_df["model"]]
        data.append(go.Bar(
                x=prompts_x,
                y=prompt_df["score"],
                name=' '.join(p.split('-')[-2:]),
                marker_color=IBM_COLORS[j],
                legendgroup=str(i),
                # barmode="group",
                )
        )
    barfig = go.Figure(data=data)
    for k in range(len(barfig.data)):
        fig.add_trace(
            barfig.data[k],
            row=i+1,
            col=1,
        )
    # fig.update_traces(marker_color=IBM_COLORS[i])

    fig.update_xaxes(
        # title_text=f"model accuracy for {dataset} prompts", row=row, col=col,
         tickangle=25
    )

    fig.update_yaxes(title_text="accuracy", row=i+1, col=1, range=[0, 1])
    fig.update_layout(legend_tracegroupgap = 40)
# print(df.head())
# fig = px.bar(
#     df,
#     x="model",
#     y="score",
#     color="dataset",
#     color_discrete_sequence=IBM_COLORS,
#     # marginal="box",  # or violin, rug
#     barmode="group",
#     # title=f"{y} distribution by split"
# )
fig.show()
