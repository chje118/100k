import pandas as pd
import ast
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp


df_figure = pd.read_csv("C:\\Users\\chris\\OneDrive\\Dokumenter\\SDU\\Master's Thesis Project\\codes.txt", sep='\t')
df_figure = df_figure.loc[:, ~df_figure.columns.str.contains('^Unnamed')]


prefix_colors = {
    'M': '#457b9d',
    'T': '#bde0fe',
    'F': '#b7e4c7',
    'P': '#ffd6a5',
    'Æ': '#ffb4a2',
    'S': '#cdb4db',
    'Other': '#ffb4a2',
    'Function': '#b7e4c7',
    'Procedure': '#ffd6a5',
    'Etiology': '#ffb4a2',
    'Disease': '#cdb4db'
}

# SNOMED Distribution Sunburst
# Combine Æ, S, and F into "Other"
df_sunburst = df_figure.copy()
df_sunburst['Prefix'] = df_sunburst['Prefix'].replace({'Æ': 'Other', 'S': 'Other', 'F': 'Other'})

fig = px.sunburst(
    df_sunburst,
    path=['Prefix', 'Category'],
    values='Count',
    color='Prefix',
    color_discrete_map=prefix_colors,
    title='Distribution of SNOMED Codes'
)
fig.update_traces(
    textinfo='label+percent entry',
    insidetextorientation='horizontal'
)
fig.show()


# Top Morphology Codes
df_m = df_figure[df_figure['Prefix'] == 'M'].copy().sort_values('Count', ascending=False).head(10)
fig_m = go.Figure()
fig_m.add_trace(go.Bar(
    y=df_m['Category'][::-1], 
    x=df_m['Count'][::-1],
    orientation='h',
    marker_color=prefix_colors['M'],
    name='Morphology',
    text=df_m['Count'][::-1],
    textposition='outside'
))
fig_m.update_layout(
    template="simple_white",
    title="Top 10 Morphology Categories",
    xaxis_title="Count",
    yaxis_title="Category",
    showlegend=False
)
fig_m.show()

# Top Topography Codes
df_t = df_figure[df_figure['Prefix'] == 'T'].copy().sort_values('Count', ascending=False).head(10)
fig_t = go.Figure()
fig_t.add_trace(go.Bar(
    y=df_t['Category'][::-1],
    x=df_t['Count'][::-1],
    orientation='h',
    marker_color=prefix_colors['T'],
    name='Topography',
    text=df_t['Count'][::-1],
    textposition='outside'
))
fig_t.update_layout(
    template="simple_white",
    title="Top 10 Topography Categories",
    xaxis_title="Count",
    yaxis_title="Category",
    showlegend=False
)
fig_t.show()

# Zoomed Sunburst for Æ, S and F
# total_value = df_figure['Count'].sum()
# df_figure['percent_total'] = df_figure['Count'] / total_value * 100

zoomed_df = df_figure[df_figure['Prefix'].isin(['S', 'F', 'Æ'])].copy()
fig_other_sunburst = px.sunburst(
    zoomed_df,
    path=['Category'],
    values='Count',
    color='Category',
    color_discrete_map=prefix_colors,
    title='Distribution of other SNOMED Codes (Æ, S, F)',
)
fig_other_sunburst.update_traces(
    textinfo='label+percent entry',
    insidetextorientation='horizontal'
)
fig_other_sunburst.show()
