from glob import glob
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def extract_root(word: str) -> str:
    if "|" in word:
        return word.split("|")[0]
    return word


def load_data(path: str) -> tuple[pd.DataFrame, str]:
    """Returns cleaned data and title of the table."""
    data = pd.read_excel(path)
    title = str(data.columns[0])
    # Remove empty rows
    data = data.dropna(axis="index", how="all")
    take_from = data[data[data.columns[0]] == "Word 1"].index[0] + 1
    data = data.iloc[take_from:]
    columns = pd.Series(data.columns)
    columns[:3] = ["word1", "word2", "effect"]
    data.columns = columns
    data.word1 = data.word1.map(extract_root)
    return data, title


def build_graph(data: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(pd.unique(pd.concat([data.word1, data.word2])))
    edge_data = data.query("effect != 0")
    edge_data = edge_data[["word1", "word2", "effect"]]
    print("Edges:")
    print(edge_data)
    print("\n")
    g.add_weighted_edges_from(edge_data.to_numpy())
    return g


def create_node_trace(graph: nx.Graph, pos: dict) -> go.Scatter:
    node_pos = []
    text = []
    for node in graph.nodes():
        x, y = pos[node]
        text.append(node)
        node_pos.append((x, y))
    node_pos = np.array(node_pos)
    node_x, node_y = node_pos.T
    return go.Scatter(
        x=node_x,
        y=node_y,
        text=text,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=16),
        marker=dict(size=12, color="grey", line=dict(width=2)),
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


FILES_PATTERN = "dat/greek/models/cooccurrance/*.xlsx"


def main():
    files = glob(FILES_PATTERN)
    for file in files:
        data, title = load_data(file)
        print(title)
        out_name = Path(file).stem
        graph = build_graph(data)
        figure = plot_network(graph)
        figure.update_layout(title=title)
        figure.write_image(
            f"figures/collocations/{out_name}_network.png", scale=2.5
        )


if __name__ == "__main__":
    main()
