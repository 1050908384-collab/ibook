import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.config import DATA_DIR
from backend.db.graph_client import GraphClient

SEEDS_DIR = DATA_DIR / "seeds"
GRAPH_DATA_DIR = DATA_DIR / "graph_data"
CURRENT_DIR = GRAPH_DATA_DIR / "current"
DERIVED_DIR = CURRENT_DIR / "derived"

BLOCKS_FILE = SEEDS_DIR / "knowledge_blocks.json"
SUBTOPICS_FILE = SEEDS_DIR / "knowledge_subtopics.json"


def _load_required_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是数组")
    return data


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_node_index(rows: list[dict], id_field: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        node_id = row[id_field]
        if node_id in index:
            raise ValueError(f"发现重复节点 ID: {node_id}")
        index[node_id] = row
    return index


def _append_edge(edges: list[dict], source_id: str, target_id: str, relation_type: str, scope: str, **extra: object) -> None:
    edges.append(
        {
            "edge_id": f"{relation_type}:{source_id}->{target_id}",
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "scope": scope,
            **extra,
        }
    )


def _topological_levels(graph: nx.DiGraph) -> dict[str, int]:
    if not graph.nodes:
        return {}
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("前置关系图存在环，无法计算学习层级")

    indegree = dict(graph.in_degree())
    levels: dict[str, int] = {}
    ready = [node for node, degree in indegree.items() if degree == 0]

    while ready:
        node = ready.pop(0)
        parent_levels = [levels[parent] for parent in graph.predecessors(node)]
        levels[node] = max(parent_levels) + 1 if parent_levels else 0
        for child in graph.successors(node):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    return levels


def _build_validation_report(
    blocks: list[dict],
    subtopics: list[dict],
    block_index: dict[str, dict],
    subtopic_index: dict[str, dict],
    edges: list[dict],
) -> dict:
    missing_block_prereqs: list[dict] = []
    missing_subtopic_prereqs: list[dict] = []

    for block in blocks:
        for prerequisite_id in block.get("prerequisites", []):
            if prerequisite_id not in block_index:
                missing_block_prereqs.append(
                    {"block_id": block["block_id"], "missing_prerequisite": prerequisite_id}
                )

    for subtopic in subtopics:
        for prerequisite_id in subtopic.get("prerequisites", []):
            if prerequisite_id not in subtopic_index:
                missing_subtopic_prereqs.append(
                    {
                        "subtopic_id": subtopic["subtopic_id"],
                        "missing_prerequisite": prerequisite_id,
                    }
                )

    prerequisite_edges = [edge for edge in edges if edge["relation_type"] == "PREREQUISITE_OF"]
    block_graph = nx.DiGraph()
    subtopic_graph = nx.DiGraph()
    block_graph.add_nodes_from(block_index)
    subtopic_graph.add_nodes_from(subtopic_index)

    for edge in prerequisite_edges:
        if edge["scope"] == "block":
            block_graph.add_edge(edge["source_id"], edge["target_id"])
        if edge["scope"] == "subtopic":
            subtopic_graph.add_edge(edge["source_id"], edge["target_id"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing_block_prerequisites": missing_block_prereqs,
        "missing_subtopic_prerequisites": missing_subtopic_prereqs,
        "block_cycles": [list(cycle) for cycle in nx.simple_cycles(block_graph)],
        "subtopic_cycles": [list(cycle) for cycle in nx.simple_cycles(subtopic_graph)],
        "isolated_blocks": sorted(list(nx.isolates(block_graph))),
        "isolated_subtopics": sorted(list(nx.isolates(subtopic_graph))),
        "edge_counts": {
            "contains": sum(edge["relation_type"] == "CONTAINS" for edge in edges),
            "prerequisite": sum(edge["relation_type"] == "PREREQUISITE_OF" for edge in edges),
            "progresses_to": sum(edge["relation_type"] == "PROGRESSES_TO" for edge in edges),
        },
    }


def _build_prerequisite_closure(
    block_index: dict[str, dict],
    subtopic_index: dict[str, dict],
    edges: list[dict],
) -> dict:
    block_graph = nx.DiGraph()
    subtopic_graph = nx.DiGraph()
    block_graph.add_nodes_from(block_index)
    subtopic_graph.add_nodes_from(subtopic_index)

    for edge in edges:
        if edge["relation_type"] != "PREREQUISITE_OF":
            continue
        if edge["scope"] == "block":
            block_graph.add_edge(edge["source_id"], edge["target_id"])
        if edge["scope"] == "subtopic":
            subtopic_graph.add_edge(edge["source_id"], edge["target_id"])

    block_closure = {
        node_id: sorted(nx.ancestors(block_graph, node_id))
        for node_id in sorted(block_index)
    }
    subtopic_closure = {
        node_id: sorted(nx.ancestors(subtopic_graph, node_id))
        for node_id in sorted(subtopic_index)
    }

    return {
        "blocks": block_closure,
        "subtopics": subtopic_closure,
    }


def _build_learning_levels(
    blocks: list[dict],
    subtopics: list[dict],
    edges: list[dict],
) -> dict:
    block_graph = nx.DiGraph()
    subtopic_graph = nx.DiGraph()

    for block in blocks:
        block_graph.add_node(block["block_id"])
    for subtopic in subtopics:
        subtopic_graph.add_node(subtopic["subtopic_id"])

    for edge in edges:
        if edge["relation_type"] != "PREREQUISITE_OF":
            continue
        if edge["scope"] == "block":
            block_graph.add_edge(edge["source_id"], edge["target_id"])
        if edge["scope"] == "subtopic":
            subtopic_graph.add_edge(edge["source_id"], edge["target_id"])

    block_levels = _topological_levels(block_graph)
    subtopic_levels = _topological_levels(subtopic_graph)

    return {
        "blocks": [
            {"block_id": block["block_id"], "level": block_levels[block["block_id"]]}
            for block in sorted(blocks, key=lambda item: item["block_id"])
        ],
        "subtopics": [
            {
                "subtopic_id": subtopic["subtopic_id"],
                "block_id": subtopic["block_id"],
                "level": subtopic_levels[subtopic["subtopic_id"]],
            }
            for subtopic in sorted(subtopics, key=lambda item: item["subtopic_id"])
        ],
    }


def build_knowledge_graph() -> dict:
    print("Building knowledge graph from seed data...")

    blocks = _load_required_json(BLOCKS_FILE)
    subtopics = _load_required_json(SUBTOPICS_FILE)
    block_index = _build_node_index(blocks, "block_id")
    subtopic_index = _build_node_index(subtopics, "subtopic_id")

    nodes: list[dict] = []
    edges: list[dict] = []
    subtopics_by_block: dict[str, list[dict]] = defaultdict(list)

    for block in blocks:
        nodes.append(
            {
                "node_id": block["block_id"],
                "node_type": "block",
                **block,
            }
        )
        for prerequisite_id in block.get("prerequisites", []):
            _append_edge(
                edges,
                prerequisite_id,
                block["block_id"],
                relation_type="PREREQUISITE_OF",
                scope="block",
                strength="required",
                source_type="seed_declared",
            )

    ordered_blocks = sorted(blocks, key=lambda item: item["block_id"])
    for current_block, next_block in zip(ordered_blocks, ordered_blocks[1:]):
        _append_edge(
            edges,
            current_block["block_id"],
            next_block["block_id"],
            relation_type="PROGRESSES_TO",
            scope="block",
            strength="ordered",
            source_type="derived_from_block_order",
        )

    for subtopic in subtopics:
        nodes.append(
            {
                "node_id": subtopic["subtopic_id"],
                "node_type": "subtopic",
                **subtopic,
            }
        )
        subtopics_by_block[subtopic["block_id"]].append(subtopic)
        _append_edge(
            edges,
            subtopic["block_id"],
            subtopic["subtopic_id"],
            relation_type="CONTAINS",
            scope="block_to_subtopic",
            strength="structural",
            source_type="seed_declared",
        )
        for prerequisite_id in subtopic.get("prerequisites", []):
            _append_edge(
                edges,
                prerequisite_id,
                subtopic["subtopic_id"],
                relation_type="PREREQUISITE_OF",
                scope="subtopic",
                strength="required",
                source_type="seed_declared",
            )

    for block_id, block_subtopics in subtopics_by_block.items():
        ordered_subtopics = sorted(block_subtopics, key=lambda item: item["sequence_index"])
        for current_subtopic, next_subtopic in zip(ordered_subtopics, ordered_subtopics[1:]):
            _append_edge(
                edges,
                current_subtopic["subtopic_id"],
                next_subtopic["subtopic_id"],
                relation_type="PROGRESSES_TO",
                scope="subtopic",
                strength="ordered",
                source_type="derived_from_sequence_index",
                block_id=block_id,
            )

    validation_report = _build_validation_report(blocks, subtopics, block_index, subtopic_index, edges)
    if validation_report["missing_block_prerequisites"] or validation_report["missing_subtopic_prerequisites"]:
        raise ValueError("seed 数据存在缺失前置引用，请先修正 JSON")
    if validation_report["block_cycles"] or validation_report["subtopic_cycles"]:
        raise ValueError("前置关系存在环，无法生成有效知识图谱")

    prerequisite_closure = _build_prerequisite_closure(block_index, subtopic_index, edges)
    learning_levels = _build_learning_levels(blocks, subtopics, edges)

    graph_client = GraphClient()
    graph_client.graph.clear()
    for node in nodes:
        node_attributes = dict(node)
        node_id = node_attributes.pop("node_id")
        graph_client.add_node(node_id, **node_attributes)
    for edge in edges:
        edge_attributes = dict(edge)
        source_id = edge_attributes.pop("source_id")
        target_id = edge_attributes.pop("target_id")
        edge_attributes.pop("edge_id")
        graph_client.add_edge(source_id, target_id, **edge_attributes)
    graph_client.save_graph()

    _write_jsonl(CURRENT_DIR / "nodes.jsonl", nodes)
    _write_jsonl(CURRENT_DIR / "edges.jsonl", edges)
    _write_json(DERIVED_DIR / "prerequisite_closure.json", prerequisite_closure)
    _write_json(DERIVED_DIR / "learning_levels.json", learning_levels)
    _write_json(DERIVED_DIR / "validation_report.json", validation_report)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed_files": {
            "blocks": str(BLOCKS_FILE.relative_to(ROOT_DIR)).replace("\\", "/"),
            "subtopics": str(SUBTOPICS_FILE.relative_to(ROOT_DIR)).replace("\\", "/"),
        },
        "node_counts": {
            "total": len(nodes),
            "blocks": len(blocks),
            "subtopics": len(subtopics),
        },
        "edge_counts": validation_report["edge_counts"],
        "runtime_graph_path": str(graph_client.graph_path.relative_to(ROOT_DIR)).replace("\\", "/"),
    }
    _write_json(CURRENT_DIR / "manifest.json", manifest)

    summary = {
        "nodes": len(nodes),
        "edges": len(edges),
        "blocks": len(blocks),
        "subtopics": len(subtopics),
        "graph_path": str(graph_client.graph_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build_knowledge_graph()
