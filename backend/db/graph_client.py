# -*- coding: utf-8 -*-
#NetworkX 知识图谱加载
import networkx as nx
import pickle
from config import settings
from pathlib import Path

class GraphClient:
    def __init__(self):
        self.graph_path = Path(settings.GRAPH_PATH)
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            self.graph = nx.DiGraph()

    def add_node(self, node_id: str, **attributes) -> None:
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, from_node: str, to_node: str, **attributes) -> None:
        self.graph.add_edge(from_node, to_node, **attributes)

    def get_prerequisites(self, node_id: str) -> list:
        return list(self.graph.predecessors(node_id))

    def get_dependents(self, node_id: str) -> list:
        return list(self.graph.successors(node_id))

    def has_path(self, from_node: str, to_node: str) -> bool:
        return nx.has_path(self.graph, from_node, to_node)

    def save_graph(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def get_all_nodes(self) -> list:
        return list(self.graph.nodes(data=True))

    def get_all_edges(self) -> list:
        return list(self.graph.edges(data=True))
