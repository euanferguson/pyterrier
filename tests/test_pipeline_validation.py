import pyterrier as pt
import unittest
import tempfile
import shutil
import pandas as pd
from pyterrier.model import PipelineError

from .base import BaseTestCase


class TestPipelineValidation(BaseTestCase):

    def test_invalid_pipeline_composition_fails(self):
        vaswani_dataset = pt.datasets.get_dataset("vaswani")
        indexref = vaswani_dataset.get_index()
        queries = vaswani_dataset.get_topics()

        # We instantiate a member of each transformer family, then we test that each invalid pipeline composition is
        # caught
        rewrite = pt.rewrite.SDM()
        retrieval = pt.BatchRetrieve(indexref, controls={"wmodel": "BM25"})
        expansion = pt.rewrite.QueryExpansion(indexref)
        reranking = pt.pipelines.PerQueryMaxMinScoreTransformer()

        rewrite_into_expansion = rewrite >> expansion
        self.assertRaises(PipelineError, rewrite_into_expansion.validate, queries)

        rewrite_into_reranking = rewrite >> reranking
        self.assertRaises(PipelineError, rewrite_into_reranking.validate, queries)

        ranked_docs = retrieval(queries)

        expansion_into_expansion = expansion >> expansion
        self.assertRaises(PipelineError, expansion_into_expansion.validate, ranked_docs)

        expansion_into_reranking = expansion >> reranking
        self.assertRaises(PipelineError, expansion_into_reranking.validate, ranked_docs)

    def test_invalid_pipeline_operators_fails(self):
        vaswani_dataset = pt.datasets.get_dataset("vaswani")
        queries = vaswani_dataset.get_topics()

        # We instantiate a member of some transformer families that we know the minimal input/output for, and use them
        # to test that invalid transformer operations are caught
        rewrite = pt.rewrite.SDM()

        set_union = rewrite | rewrite
        self.assertRaises(PipelineError, set_union.validate, queries)

        set_intersection = rewrite & rewrite
        self.assertRaises(PipelineError, set_intersection.validate, queries)

        feature_union = rewrite ** rewrite
        self.assertRaises(PipelineError, feature_union.validate, queries)

        comb_sum = rewrite + rewrite
        self.assertRaises(PipelineError, comb_sum.validate, queries)

        concatenate = rewrite ^ rewrite
        self.assertRaises(PipelineError, concatenate.validate, queries)

        ranked_cutoff = rewrite % 10
        self.assertRaises(PipelineError, ranked_cutoff.validate, queries)

if __name__ == "__main__":
    unittest.main()