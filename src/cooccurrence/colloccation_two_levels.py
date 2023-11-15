import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_graph(data: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    all_words = pd.unique(pd.concat([data.word1, data.word2]))
    seed_words = data.query("level == 1")
    seed_words = pd.concat([seed_words.word1, seed_words.word2])
    seed_words = set(seed_words)
    g.add_nodes_from(all_words)
    data = data[["word1", "word2", "effect"]]
    g.add_weighted_edges_from(data.to_numpy())
    is_seed = {word: word in seed_words for word in all_words}
    nx.set_node_attributes(g, is_seed, "is_seed")
    return g


def create_node_trace(graph: nx.Graph, pos: dict) -> go.Scatter:
    node_pos = []
    text = []
    color = []
    for node, is_seed in graph.nodes(data="is_seed", default=False):
        x, y = pos[node]
        text.append(node)
        node_pos.append((x, y))
        if is_seed:
            color.append("orange")
        else:
            color.append("grey")
    node_pos = np.array(node_pos)
    node_x, node_y = node_pos.T
    return go.Scatter(
        x=node_x,
        y=node_y,
        text=text,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=16),
        marker=dict(size=12, color=color, line=dict(width=2)),
    )


def create_edge_trace(graph: nx.Graph, pos: dict) -> go.Scatter:
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="black", width=1.0),
        opacity=0.4,
    )


def plot_network(graph: nx.Graph) -> go.Figure:
    pos = nx.spring_layout(graph)
    nodes = create_node_trace(graph, pos)
    edges = create_edge_trace(graph, pos)
    fig = go.Figure([edges, nodes])
    fig = fig.update_layout(showlegend=False, template="plotly_white")
    fig = fig.update_xaxes(showticklabels=False)
    fig = fig.update_yaxes(showticklabels=False)
    return fig


def plot_heatmap(graph: nx.Graph) -> go.Figure:
    node_names = list(graph.nodes)
    adjacency = pd.DataFrame(
        nx.adjacency_matrix(graph).todense(),
        columns=node_names,
        index=node_names,
    )
    return px.imshow(adjacency).update_xaxes(side="top")


filon = pd.read_csv("dat/filon_colloc.csv")
graph = build_graph(filon)
fig = plot_network(graph)
fig.update_layout(title="Collocations in Filon", width=1000, height=1000)
fig.write_image("figures/collocations/filon_layered.png", scale=2.5)
fig.show()

fig = plot_heatmap(build_graph(filon.query("level==1")))
fig.update_layout(title="Collocations in Filon", width=600, height=600)
fig.write_image("figures/collocations/filon_heatmap.png", scale=2.5)
fig.show()

plutarch = pd.read_csv("dat/plutarch_colloc.csv")
graph = build_graph(plutarch)
fig = plot_network(graph)
fig.update_layout(title="Collocations in Plutarch", width=1000, height=1000)
fig.write_image("figures/collocations/plutarch_layered.png", scale=2.5)
fig.show()

fig = plot_heatmap(build_graph(plutarch.query("level==1")))
fig.update_layout(title="Collocations in Plutarch", width=600, height=600)
fig.write_image("figures/collocations/plutarch_heatmap.png", scale=2.5)
fig.show()
