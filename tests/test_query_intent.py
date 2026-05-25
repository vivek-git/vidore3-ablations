from __future__ import annotations

import unittest

from vidore3_ablations.query_intent import QueryIntentGraphBuilder


class QueryIntentGraphBuilderTest(unittest.TestCase):
    def test_builds_nodes_and_edges_for_significant_terms(self) -> None:
        graph = QueryIntentGraphBuilder().build(
            "q1",
            "Locate FUEL pressure table 3",
        )

        nodes_by_text = {node.text: node for node in graph.nodes}
        self.assertEqual(nodes_by_text["locate"].kind, "action")
        self.assertEqual(nodes_by_text["fuel"].kind, "identifier")
        self.assertEqual(nodes_by_text["pressure"].kind, "attribute")
        self.assertEqual(nodes_by_text["3"].kind, "value")
        self.assertIn("fuel pressure", graph.edge_phrases())
        self.assertTrue(any(edge.relation == "requests" for edge in graph.edges))


class QueryIntentGraphBM25PipelineTest(unittest.TestCase):
    def test_graph_pipeline_ranks_intent_matching_document_first(self) -> None:
        try:
            from vidore3_ablations.pipelines.query_intent_graph import (
                QueryIntentGraphBM25Pipeline,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"optional retrieval dependencies unavailable: {exc}")

        pipeline = QueryIntentGraphBM25Pipeline(top_k=2)
        pipeline.index(
            corpus_ids=["doc-fuel", "doc-wiring"],
            corpus_images=[object(), object()],
            corpus_texts=[
                "Fuel pressure table lists the operating limit.",
                "Electrical wiring diagram and connector locations.",
            ],
        )

        results = pipeline.search(["q1"], ["Locate fuel pressure table"])
        if isinstance(results, tuple):
            run, infos = results
        else:
            run, infos = results, {}

        self.assertEqual(next(iter(run["q1"])), "doc-fuel")
        self.assertIn("query_intent_graphs", infos)


if __name__ == "__main__":
    unittest.main()
