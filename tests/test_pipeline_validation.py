import pyterrier as pt
import unittest
import tempfile
import shutil
import pandas as pd
from pyterrier.model import PipelineError
from pyterrier.pipelines import PerQueryMaxMinScoreTransformer

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

    def test_valid_pipelines_pass(self):
        vaswani_dataset = pt.datasets.get_dataset("vaswani")
        indexref = vaswani_dataset.get_index()
        queries = vaswani_dataset.get_topics()

        # We instantiate the possible valid pipelines according to the transformer families and validate them
        rewrite = pt.rewrite.SDM()
        retrieval = pt.BatchRetrieve(indexref, controls={"wmodel": "BM25"})
        expansion = pt.rewrite.QueryExpansion(indexref)
        reranking = pt.pipelines.PerQueryMaxMinScoreTransformer()
        featurescoring = pt.FeaturesBatchRetrieve(indexref, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])
        ranked_docs = retrieval(queries)

        # Matching minimal input and minimal output
        rewrite_into_retrieval = rewrite >> retrieval
        rewrite_into_retrieval.validate(queries)

        rewrite_into_feature_scoring = rewrite >> featurescoring
        rewrite_into_feature_scoring.validate(queries)

        expansion_into_retrieval = expansion >> retrieval
        expansion_into_retrieval.validate(ranked_docs)

        expansion_into_feature_scoring = expansion >> featurescoring
        expansion_into_feature_scoring.validate(ranked_docs)

        retrieval_into_expansion = retrieval >> expansion
        retrieval_into_expansion.validate(queries)

        retrieval_into_reranking = retrieval >> reranking
        retrieval_into_reranking.validate(queries)

        reranking_into_expansion = reranking >> expansion
        reranking_into_expansion.validate(ranked_docs)

        reranking_into_reranking = reranking >> reranking
        reranking_into_reranking.validate(ranked_docs)

        # minimal output is superset of minimal input
        retrieval_into_rewrite = retrieval >> rewrite
        retrieval_into_rewrite.validate(queries)

        retrieval_into_retrieval = retrieval >> retrieval
        retrieval_into_retrieval.validate(queries)

        reranking_into_rewrite = reranking >> rewrite
        reranking_into_rewrite.validate(ranked_docs)

        reranking_into_retrieval = reranking >> retrieval
        reranking_into_retrieval.validate(ranked_docs)

    def test_feature_scoring_validates(self):
        vaswani_dataset = pt.datasets.get_dataset("vaswani")
        indexref = vaswani_dataset.get_index()
        queries = vaswani_dataset.get_topics()

        # We instantiate the possible valid pipelines according to the transformer families and validate them
        rewrite = pt.rewrite.SDM()
        retrieval = pt.BatchRetrieve(indexref, controls={"wmodel": "BM25"})
        expansion = pt.rewrite.QueryExpansion(indexref)
        reranking = pt.pipelines.PerQueryMaxMinScoreTransformer()
        featurescoring = pt.FeaturesBatchRetrieve(indexref, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])
        ranked_docs = retrieval(queries)

        feature_scoring_into_rewrite = featurescoring >> rewrite
        feature_scoring_into_rewrite.validate(queries)

        feature_scoring_into_retrieval = featurescoring >> retrieval
        feature_scoring_into_retrieval.validate(queries)

        feature_scoring_into_expansion = featurescoring >> expansion
        feature_scoring_into_expansion.validate(ranked_docs)

        feature_scoring_into_reranking = featurescoring >> reranking
        feature_scoring_into_reranking.validate(ranked_docs)


    def test_validate_returns_correct_columns(self):
        # For some example pipelines given in the PyTerrier docs, we test that when validating the correct columns are
        # returned
        dataset = pt.datasets.get_dataset("vaswani")
        indexref = dataset.get_index()

        BM25 = pt.BatchRetrieve(indexref, controls={"wmodel": "BM25"})
        PL2 = pt.BatchRetrieve(indexref, controls={"wmodel": "PL2"})
        topics = dataset.get_topics()

        validate_output = BM25.validate(topics)
        transform_output = BM25.transform(topics)
        self.assertSetEqual(set(validate_output), set(transform_output.columns))

        res_union = BM25 | PL2
        validate_output = res_union.validate(topics)
        transform_output = res_union.transform(topics)
        self.assertSetEqual(set(validate_output), set(transform_output.columns))

        # # TODO: Test set intersection on newest version
        # res_union = BM25 & PL2
        # validate_output = res_union.validate(topics)
        # transform_output = res_union.transform(topics)
        # self.assertSetEqual(set(validate_output), set(transform_output.columns))

        reranker = ((BM25 % 100 >> PerQueryMaxMinScoreTransformer()) ^ BM25) % 1000
        validate_output = reranker.validate(topics)
        transform_output = reranker.transform(topics)
        self.assertSetEqual(set(validate_output), set(transform_output.columns))

        # TODO: Test linear combination on newest version
        # bm25 = pt.BatchRetrieve(indexref, wmodel="BM25") >> PerQueryMaxMinScoreTransformer()
        # dph = pt.BatchRetrieve(indexref, wmodel="DPH") >> PerQueryMaxMinScoreTransformer()
        # linear_combine = 0.75 * bm25 + 0.25 * dph
        # validate_output = linear_combine.validate(topics)
        # transform_output = linear_combine.transform(topics)
        # self.assertSetEqual(set(validate_output), set(transform_output.columns))

        # TODO: Test feature union pipeline on newest version
        # pipe = BM25 >> (TF_IDF ** PL2)
        # validate_output = pipe.validate("chemical end:2")
        # transform_output = pipe.transform("chemical end:2")
        # self.assertSetEqual(set(validate_output), set(transform_output.columns))

if __name__ == "__main__":
    unittest.main()