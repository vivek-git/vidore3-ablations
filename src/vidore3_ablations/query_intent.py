"""Query intent graph construction for retrieval pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

ACTION_TERMS = {
    "calculate",
    "check",
    "compare",
    "define",
    "describe",
    "determine",
    "explain",
    "find",
    "identify",
    "inspect",
    "list",
    "locate",
    "name",
    "provide",
    "show",
}

ATTRIBUTE_TERMS = {
    "amount",
    "capacity",
    "code",
    "diagram",
    "figure",
    "limit",
    "location",
    "number",
    "pressure",
    "procedure",
    "section",
    "step",
    "table",
    "temperature",
    "value",
    "warning",
}


@dataclass(frozen=True)
class QueryIntentNode:
    """A query concept or action extracted from natural language."""

    id: str
    text: str
    kind: str
    weight: float


@dataclass(frozen=True)
class QueryIntentEdge:
    """Relationship between two query intent nodes."""

    source: str
    target: str
    relation: str
    text: str
    weight: float


@dataclass(frozen=True)
class QueryIntentGraph:
    """Lightweight graph representation of the retrieval intent in a query."""

    query_id: str
    raw_query: str
    nodes: List[QueryIntentNode]
    edges: List[QueryIntentEdge]

    def node_terms(self) -> List[str]:
        return [node.text for node in self.nodes]

    def edge_phrases(self) -> List[str]:
        return [edge.text for edge in self.edges if edge.text]

    def to_dict(self) -> Dict[str, object]:
        return {
            "query_id": self.query_id,
            "raw_query": self.raw_query,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
        }


def normalize_text(text: str) -> str:
    """Normalize text for lexical matching while preserving token order."""

    return " ".join(token.lower() for token in TOKEN_RE.findall(text or ""))


def tokenize_intent_text(text: str) -> List[str]:
    """Tokenize text with the same normalization used by intent graphs."""

    return TOKEN_RE.findall(text.lower() if text else "")


class QueryIntentGraphBuilder:
    """Build deterministic query intent graphs without model dependencies."""

    def build(self, query_id: str, query: str) -> QueryIntentGraph:
        tokens = TOKEN_RE.findall(query or "")
        significant = [
            (idx, token, token.lower())
            for idx, token in enumerate(tokens)
            if token.lower() not in STOPWORDS
        ]

        nodes: List[QueryIntentNode] = []
        token_to_node_id: Dict[int, str] = {}
        seen: Dict[str, str] = {}
        for token_idx, token, normalized in significant:
            if normalized in seen:
                token_to_node_id[token_idx] = seen[normalized]
                continue

            node_id = f"n{len(nodes)}"
            seen[normalized] = node_id
            token_to_node_id[token_idx] = node_id
            nodes.append(
                QueryIntentNode(
                    id=node_id,
                    text=normalized,
                    kind=self._classify(token),
                    weight=self._weight(token),
                )
            )

        edges = self._build_edges(significant, token_to_node_id, nodes)
        return QueryIntentGraph(
            query_id=query_id,
            raw_query=query,
            nodes=nodes,
            edges=edges,
        )

    def build_many(
        self,
        query_ids: Sequence[str],
        queries: Sequence[str],
    ) -> Dict[str, QueryIntentGraph]:
        return {
            query_id: self.build(query_id, query)
            for query_id, query in zip(query_ids, queries)
        }

    def _classify(self, token: str) -> str:
        normalized = token.lower()
        if normalized in ACTION_TERMS:
            return "action"
        if normalized in ATTRIBUTE_TERMS:
            return "attribute"
        if any(char.isdigit() for char in token):
            return "value"
        if token.isupper() and len(token) > 1:
            return "identifier"
        return "concept"

    def _weight(self, token: str) -> float:
        kind = self._classify(token)
        weight_by_kind = {
            "action": 1.15,
            "attribute": 1.1,
            "concept": 1.0,
            "identifier": 1.25,
            "value": 1.2,
        }
        return weight_by_kind[kind]

    def _build_edges(
        self,
        significant: Sequence[tuple[int, str, str]],
        token_to_node_id: Mapping[int, str],
        nodes: Sequence[QueryIntentNode],
    ) -> List[QueryIntentEdge]:
        edges: List[QueryIntentEdge] = []
        node_by_id = {node.id: node for node in nodes}

        for left, right in zip(significant, significant[1:]):
            left_idx, _, left_text = left
            right_idx, _, right_text = right
            source = token_to_node_id[left_idx]
            target = token_to_node_id[right_idx]
            if source == target:
                continue
            token_gap = right_idx - left_idx
            relation = "adjacent_context" if token_gap == 1 else "near_context"
            weight = 1.0 if token_gap == 1 else 0.75
            edges.append(
                QueryIntentEdge(
                    source=source,
                    target=target,
                    relation=relation,
                    text=f"{left_text} {right_text}",
                    weight=weight,
                )
            )

        action_nodes = [node for node in nodes if node.kind == "action"]
        if action_nodes:
            action = action_nodes[0]
            for node in nodes:
                if node.id == action.id:
                    continue
                edges.append(
                    QueryIntentEdge(
                        source=action.id,
                        target=node.id,
                        relation="requests",
                        text=f"{action.text} {node.text}",
                        weight=0.5 * node_by_id[node.id].weight,
                    )
                )

        return _dedupe_edges(edges)


def _dedupe_edges(edges: Iterable[QueryIntentEdge]) -> List[QueryIntentEdge]:
    deduped: Dict[tuple[str, str, str, str], QueryIntentEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.relation, edge.text)
        current = deduped.get(key)
        if current is None or edge.weight > current.weight:
            deduped[key] = edge
    return list(deduped.values())
