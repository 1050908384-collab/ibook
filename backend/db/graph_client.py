# -*- coding: utf-8 -*-
# NetworkX 知识图谱加载
import pickle
from pathlib import Path

import networkx as nx

from config import BACKEND_DIR, settings

class GraphClient:
    def __init__(self):
        self.graph_path = self._resolve_graph_path(settings.GRAPH_PATH)
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            self.graph = nx.DiGraph()

    @staticmethod
    def _resolve_graph_path(graph_path: str) -> Path:
        path = Path(graph_path)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return path

    def add_node(self, node_id: str, **attributes) -> None:
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, from_node: str, to_node: str, **attributes) -> None:
        relation_type = attributes.get("relation_type")
        if self.graph.has_edge(from_node, to_node):
            existing_attributes = dict(self.graph.edges[from_node, to_node])
            relation_types = existing_attributes.get("relation_types", [])
            if not relation_types and existing_attributes.get("relation_type"):
                relation_types = [existing_attributes["relation_type"]]
            if relation_type and relation_type not in relation_types:
                relation_types.append(relation_type)
            existing_attributes.update(attributes)
            if relation_types:
                existing_attributes["relation_types"] = relation_types
            self.graph.add_edge(from_node, to_node, **existing_attributes)
            return
        if relation_type:
            attributes["relation_types"] = [relation_type]
        self.graph.add_edge(from_node, to_node, **attributes)

    @staticmethod
    def _has_relation(edge_attributes: dict, relation_type: str) -> bool:
        if edge_attributes.get("relation_type") == relation_type:
            return True
        return relation_type in edge_attributes.get("relation_types", [])

    def get_prerequisites(self, node_id: str) -> list:
        return [
            predecessor
            for predecessor in self.graph.predecessors(node_id)
            if self._has_relation(self.graph.edges[predecessor, node_id], "PREREQUISITE_OF")
        ]

    def get_dependents(self, node_id: str) -> list:
        return [
            successor
            for successor in self.graph.successors(node_id)
            if self._has_relation(self.graph.edges[node_id, successor], "PREREQUISITE_OF")
        ]

    def get_contained_subtopics(self, block_id: str) -> list:
        return [
            successor
            for successor in self.graph.successors(block_id)
            if self._has_relation(self.graph.edges[block_id, successor], "CONTAINS")
        ]

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
