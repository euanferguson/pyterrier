
import types
from matchpy import ReplacementRule, Wildcard, Symbol, Operation, Arity, replace_all, Pattern, CustomConstraint
from warnings import warn
import pandas as pd
from .model import add_ranks, coerce_queries_dataframe, PipelineError, TRANSFORMER_FAMILY, TYPE_SAFETY_LEVEL,\
    QUERIES, DOCS, RANKED_DOCS, RETRIEVAL, RERANKING, QUERY_EXPANSION, QUERY_REWRITE, FEATURE_SCORING
from . import tqdm
import deprecation

LAMBDA = lambda:0
def is_lambda(v):
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
       
def is_transformer(v):
    if isinstance(v, TransformerBase):
        return True
    return False

def get_transformer(v):
    """ 
        Used to coerce functions, lambdas etc into transformers 
    """

    if isinstance(v, Wildcard):
        # get out of jail for matchpy
        return v
    if is_transformer(v):
        return v
    if is_lambda(v):
        return ApplyGenericTransformer(v)
    if isinstance(v, types.FunctionType):
        return ApplyGenericTransformer(v)
    if isinstance(v, pd.DataFrame):
        return SourceTransformer(v)
    raise ValueError("Passed parameter %s of type %s cannot be coerced into a transformer" % (str(v), type(v)))

rewrites_setup = False
rewrite_rules = []

def setup_rewrites():
    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    #three arbitrary "things".
    x = Wildcard.dot('x')
    xs = Wildcard.plus('xs')
    y = Wildcard.dot('y')
    z = Wildcard.dot('z')
    # two different match retrives
    _br1 = Wildcard.symbol('_br1', BatchRetrieve)
    _br2 = Wildcard.symbol('_br2', BatchRetrieve)
    _fbr = Wildcard.symbol('_fbr', FeaturesBatchRetrieve)
    
    # batch retrieves for the same index
    BR_index_matches = CustomConstraint(lambda _br1, _br2: _br1.indexref == _br2.indexref)
    BR_FBR_index_matches = CustomConstraint(lambda _br1, _fbr: _br1.indexref == _fbr.indexref)
    
    # rewrite nested binary feature unions into one single polyadic feature union
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(x, FeatureUnionPipeline(y,z)) ),
        lambda x, y, z: FeatureUnionPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(FeatureUnionPipeline(x,y), z) ),
        lambda x, y, z: FeatureUnionPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(FeatureUnionPipeline(x,y), xs) ),
        lambda x, y, xs: FeatureUnionPipeline(*[x,y]+list(xs))
    ))

    # rewrite nested binary compose into one single polyadic compose
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(x, ComposedPipeline(y,z)) ),
        lambda x, y, z: ComposedPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(ComposedPipeline(x,y), z) ),
        lambda x, y, z: ComposedPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(ComposedPipeline(x,y), xs) ),
        lambda x, y, xs: ComposedPipeline(*[x,y]+list(xs))
    ))

    # rewrite batch a feature union of BRs into an FBR
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(_br1, _br2), BR_index_matches),
        lambda _br1, _br2: FeaturesBatchRetrieve(_br1.indexref, ["WMODEL:" + _br1.controls["wmodel"], "WMODEL:" + _br2.controls["wmodel"]])
    ))

    def push_fbr_earlier(_br1, _fbr):
        #TODO copy more attributes
        _fbr.controls["wmodel"] = _br1.controls["wmodel"]
        return _fbr

    # rewrite a BR followed by a FBR into a FBR
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(_br1, _fbr), BR_FBR_index_matches),
        push_fbr_earlier
    ))

    global rewrites_setup
    rewrites_setup = True


class Scalar(Symbol):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value


class TransformerBase:
    """
        Base class for all transformers. Implements the various operators >> + * | & 
        as well as the compile() for rewriting complex pipelines into more simples ones.
    """

    def __init__(self, **kwargs):
        """
            When initialising a transformer we can pass in certain value in order to allow for type checking and
            pipeline validation

            Arguments:
             - family : For common transformers, the family can be passed as a constant, allowing for the default
                    data transformer family types defined in models.py to be used
            - minimal_input : The list of columns a transformer needs for its transform method to work
            - minimal_output : The list of columns that a transformer will always output from its transform method
            - true_output : The list of columns that a transformer actually outputs. This is needed for pipeline
                    validation, so it is possible to validate the input of one transformer using the output of previous
        """
        self.family = kwargs.get('family')

        # If family is defined, then we can obtain the minimal input and output from default family mapping
        if self.family:
            self.minimal_input = TRANSFORMER_FAMILY[self.family]['minimal_input']
            self.minimal_output = TRANSFORMER_FAMILY[self.family]['minimal_output']
        else:
            self.minimal_input = kwargs.get('minimal_input')
            self.minimal_output = kwargs.get('minimal_output')

        # Use type safety level to judge how strictly we need input/output to be defined
        if not hasattr(self, 'minimal_input'):
            if TYPE_SAFETY_LEVEL == 1:
                print("WARNING: Minimal input not defined for transformer " + repr(self))
            elif TYPE_SAFETY_LEVEL == 2:
                raise TypeError("Minimal input not defined for transformer " + repr(self))
        if not hasattr(self, 'minimal_output'):
            if TYPE_SAFETY_LEVEL == 1:
                print("WARNING: Minimal output not defined for transformer " + repr(self))
            elif TYPE_SAFETY_LEVEL == 2:
                raise TypeError("Minimal output not defined for transformer " + repr(self))

        self.true_output = kwargs.get('true_output')
        if not hasattr(self, 'true_output'):
            if TYPE_SAFETY_LEVEL == 1:
                print("WARNING: True output not defined for transformer " + repr(self))
            elif TYPE_SAFETY_LEVEL == 2:
                raise TypeError("True output not defined for transformer " + repr(self))

    def transform(self, topics_or_res):
        """
            Abstract method for all transformations. Typically takes as input a Pandas
            DataFrame, and also returns one.
        """
        pass
        
    def transform_gen(self, input, batch_size=1):
        """
            Method for executing a transformer pipeline on smaller batches of queries.
            The input dataframe is grouped into batches of batch_size queries, and a generator
            returned, such that transform() is only executed for a smaller batch at a time. 
        """
        docno_provided = "docno" in input.columns
        docid_provided = "docid" in input.columns
        
        if docno_provided or docid_provided:
            queries = input[["qid"]].drop_duplicates()
        else:
            queries = input
        batch=[]      
        for query in queries.itertuples():
            if len(batch) == batch_size:
                batch_topics = pd.concat(batch)
                batch=[]
                yield self.transform(batch_topics)
            batch.append(input[input["qid"] == query.qid])
        if len(batch) > 0:
            batch_topics = pd.concat(batch)
            yield self.transform(batch_topics)

    def search(self, query : str, qid : str = "1", sort=True):
        """
            Method for executing a transformer (pipeline) for a single query. 
            Returns a dataframe with the results for the specified query. This
            is a utility method, and most uses are expected to use the transform()
            method passing a dataframe.

            Arguments:
             - query(str): String form of the query to run
             - qid(str): the query id to associate to this request. defaults to 1.
             - sort(bool): ensures the results are sorted by descending rank (defaults to True)

            Example::

                bm25 = pt.BatchRetrieve(index, wmodel="BM25")
                res = bm25.search("example query")

                # is equivalent to
                queryDf = pd.DataFrame([["1", "example query"]], columns=["qid", "query"])
                res = bm25.transform(queryDf)
            
            
        """
        import pandas as pd
        queryDf = pd.DataFrame([[qid, query]], columns=["qid", "query"])
        rtr = self.transform(queryDf)
        if "qid" in rtr.columns and "rank" in rtr.columns:
            rtr = rtr.sort_values(["qid", "rank"], ascending=[True,True])
        return rtr

    def validate(self, inputs):
        '''
            Default method implementation to validate transformer types. Checks that the input dataframe to the
            transformer has the required attributes, and returns the attributes that will be provided if applicable
        '''
        if type(inputs) == str:
            # If we are given a str, we treat it as a query
            inputs = ["qid", "query"]
        elif isinstance(inputs, pd.DataFrame):
            inputs = inputs.columns

        # We are validating that the set of input columns is a superset of the set of minimal input columns
        # i.e. all required columns are present
        if set(inputs).issuperset(set(self.minimal_input)):
            return self._calculate_output(inputs)
        else:
            raise TypeError("Could not validate transformer with given input")

    def _calculate_output(self, inputs):
        '''
            Method for calculating the output columns of a transformer

            We have 4 possible cases:
                true_output == "minimal_output" - The transformer returns exactly the minimal output columns
                true_output == "input" - All input columns are passed through the transformer
                true_output is a function - The output columns must be calculated using the input columns
                true_output is a list of columns - The output columns are defined literally
        '''
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.columns

        if self.true_output == "minimal_output":
            # Transformer returns exactly the minimal output
            return self.minimal_output
        if self.true_output == "input":
            # Transformer returns all input columns, along with the minimal output
            # mathematically that is the union of the set of input columns and minimal output columns
            return list(set(inputs) | set(self.minimal_output))
        if isinstance(self.true_output, types.FunctionType):
            # Transformer requires some further calculation of return columns
            return self.true_output(inputs)
        # Else we return the specified true output
        return self.true_output

    def compile(self):
        """
            Rewrites this pipeline by applying of the Matchpy rules in rewrite_rules. Pipeline
            optimisation is discussed in the `ICTIR 2020 paper on PyTerrier <https://arxiv.org/abs/2007.14271>`_.
        """
        if not rewrites_setup:
            setup_rewrites()
        print("Applying %d rules" % len(rewrite_rules))
        return replace_all(self, rewrite_rules)

    def __call__(self, *args, **kwargs):
        """
            Sets up a default method for every transformer, which is aliased to transform(). 
        """
        return self.transform(*args, **kwargs)

    def __rshift__(self, right):
        return ComposedPipeline(self, right)

    def __rrshift__(self, left):
        return ComposedPipeline(left, self)

    def __add__(self, right):
        return CombSumTransformer(self, right)

    def __pow__(self, right):
        return FeatureUnionPipeline(self, right)

    def __mul__(self, rhs):
        assert isinstance(rhs, int) or isinstance(rhs, float)
        return ScalarProductTransformer(self, rhs)

    def __rmul__(self, lhs):
        assert isinstance(lhs, int) or isinstance(lhs, float)
        return ScalarProductTransformer(self, lhs)

    def __or__(self, right):
        return SetUnionTransformer(self, right)

    def __and__(self, right):
        return SetIntersectionTransformer(self, right)

    def __mod__(self, right):
        assert isinstance(right, int)
        return RankCutoffTransformer(self, right)

    def __xor__(self, right):
        return ConcatenateTransformer(self, right)

    def __invert__(self):
        from .cache import ChestCacheTransformer
        return ChestCacheTransformer(self)

        

    
class EstimatorBase(TransformerBase):
    """
        This is a base class for things that can be fitted.
    """
    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va):
        """
            Method for training the transformer.
            Arguments:
            - topics_or_res_tr(DataFrame): training topics (probably with documents)
            - qrels_tr(DataFrame): training qrels
            - topics_or_res_va(DataFrame): validation topics (probably with documents)
            - qrels_va(DataFrame): validation qrels
        """
        pass

class IdentityTransformer(TransformerBase, Operation):
    """
        A transformer that returns exactly the same as its input.
    """
    arity = Arity.nullary

    def __init__(self, *args, **kwargs):
        Operation.__init__(self, *args, **kwargs)

    def validate(self, inputs):
        # Identity transformer always valid
        return list(inputs.columns)

    def transform(self, topics):
        return topics

class SourceTransformer(TransformerBase, Operation):
    """
    A Transformer that can be used when results have been saved in a dataframe.
    It will select results on qid.
    If a query column is in the dataframe passed in the constructor, this will override any query
    column in the topics dataframe passed to the transform() method.
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        minimal_input = QUERIES
        minimal_output = ["qid", "query_x", "query_y"]

        def true_output(inputs):
            if isinstance(inputs, pd.DataFrame):
                inputs = inputs.columns
            # We return all columns in minimal output, plus any columns in self.df that are also in input columns
            return list(set(self.minimal_output) | (set(self.df.columns) & set(inputs)))

        super().__init__(operands=[], **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)
        self.operands=[]
        self.df = rtr[0]
        self.df_contains_query = "query" in self.df.columns
        assert "qid" in self.df.columns


    def transform(self, topics):
        assert "qid" in topics.columns
        columns=["qid"]
        topics_contains_query = "query" in topics.columns
        if not self.df_contains_query and topics_contains_query:
            columns.append("query")
        rtr = topics[columns].merge(self.df, on="qid")
        return rtr

class UniformTransformer(TransformerBase, Operation):
    """
        A transformer that returns the same dataframe every time transform()
        is called. This class is useful for testing. 
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        minimal_input = []
        minimal_output = list(rtr[0].columns)
        true_output = "minimal_output"
        Operation.__init__(self, operands=[], **kwargs)
        TransformerBase.__init__(self, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output, **kwargs)
        self.operands=[]
        self.rtr = rtr[0]
    
    def transform(self, topics):
        rtr = self.rtr.copy()
        return rtr

class BinaryTransformerBase(TransformerBase,Operation):
    """
        A base class for all operator transformers that can combine the input of exactly 2 transformers. 
    """
    arity = Arity.binary

    def __init__(self, operands, **kwargs):
        assert 2 == len(operands)
        Operation.__init__(self, operands=operands)
        TransformerBase.__init__(self, **kwargs)
        self.left = operands[0]
        self.right = operands[1]

    def validate(self, inputs):
        # validate left component
        try:
            left_output = self.left.validate(inputs)
        except TypeError:
            raise PipelineError(self.left, inputs)
        if not set(self.minimal_input).issubset(left_output):
            raise PipelineError(self, left_output, self.left)

        # validate right component
        if not isinstance(self.right, int) and not isinstance(self.right, float) and not isinstance(self.right, Scalar):
            try:
                right_output = self.right.validate(inputs)
            except TypeError:
                raise PipelineError(self.right, inputs)
            if not set(self.minimal_input).issubset(right_output):
                raise PipelineError(self, right_output, self.right)

        return self._calculate_output(left_output)


class NAryTransformerBase(TransformerBase,Operation):
    """
        A base class for all operator transformers that can combine the input of 2 or more transformers. 
    """
    arity = Arity.polyadic

    def __init__(self, operands, minimal_input=None, **kwargs):
        if minimal_input:
            TransformerBase.__init__(self, minimal_input=minimal_input)
        Operation.__init__(self, operands=operands)
        models = operands
        self.models = list( map(lambda x : get_transformer(x), models) )

    def __getitem__(self, number):
        """
            Allows access to the ith transformer.
        """
        return self.models[number]

    def __len__(self):
        """
            Returns the number of transformers in the operator.
        """
        return len(self.models)

    def validate(self, inputs):
        # We validate the first transformer in the pipeline
        try:
            next_input = self.models[0].validate(inputs)
        except TypeError:
            raise PipelineError(self, inputs, self.models[0])

        # In the case where the first transformer of an nary transformer must return a certain type, regardless of
        # further transformers
        if hasattr(self, "minimal_input"):
            try:
                super().validate(next_input)
            except TypeError:
                raise PipelineError(self, next_input, self.models[0])

        # We can then validate the other transfromer in pipeline, with using the previous transformer output
        for i in range(len(self)-1):
            try:
                next_input = self.models[i+1].validate(next_input)
            except TypeError:
                raise PipelineError(self.models[i+1], next_input, self.models[i])

        return next_input

class SetUnionTransformer(BinaryTransformerBase):
    """      
        This operator makes a retrieval set that includes documents that occur in the union (either) of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [1, "text1", doc1, 0.42] and right = [1, "text1", doc2, 0.24]. 
        Then, left | right will be a dataframe with only the columns [qid, query, docno] and two rows = [[1, "text1", doc1], [1, "text1", doc2]].
                
        In case of duplicated both containing (qid, docno), only the first occurrence will be used.
    """
    name = "Union"

    def __init__(self, operands, **kwargs):
        minimal_input = DOCS
        minimal_output = DOCS

        def true_output(inputs):
            # We return all input columns except 'score' and 'rank'
            return list(set(inputs) - {'score', 'rank'})
        super().__init__(operands=operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)


    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        import pandas as pd
        assert isinstance(res1, pd.DataFrame)
        assert isinstance(res2, pd.DataFrame)
        rtr = pd.concat([res1, res2])
        
        on_cols = ["qid", "docno"]     
        rtr = rtr.drop_duplicates(subset=on_cols)
        rtr = rtr.sort_values(by=on_cols)
        rtr.drop(columns=["score", "rank"], inplace=True, errors='ignore')
        return rtr

class SetIntersectionTransformer(BinaryTransformerBase):
    """
        This operator makes a retrieval set that only includes documents that occur in the intersection of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [[1, "text1", doc1, 0.42]] (one row) and right = [[1, "text1", doc1, 0.24],[1, "text1", doc2, 0.24]] (two rows).
        Then, left & right will be a dataframe with only the columns [qid, query, docno] and one single row = [[1, "text1", doc1]].
                
        For columns other than (qid, docno), only the left value will be used.
    """
    name = "Intersect"

    def __init__(self, operands, **kwargs):
        minimal_input = DOCS
        minimal_output = DOCS

        def true_output(inputs):
            # We return all input columns except 'score' and 'rank'
            return set(inputs) - {'score', 'rank'}

        super().__init__(operands=operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)  
        
        on_cols = ["qid", "docno"]
        rtr = res1.merge(res2, on=on_cols, suffixes=('','_y'))
        rtr.drop(columns=["score", "rank"], inplace=True, errors='ignore')
        return rtr

class CombSumTransformer(BinaryTransformerBase):
    """
        Adds the scores of documents from two different retrieval transformers.
        Documents not present in one transformer are given a score of 0.
    """
    name = "Sum"

    def __init__(self, operands, **kwargs):
        minimal_input = RANKED_DOCS
        minimal_output = RANKED_DOCS
        true_output = "input"
        super().__init__(operands=operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)

    def transform(self, topics_and_res):
        res1 = self.left.transform(topics_and_res)
        res2 = self.right.transform(topics_and_res)
        both_cols = set(res1.columns) & set(res2.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        merged = res1.merge(res2, on=["qid", "docno"], suffixes=[None, "_r"])
        merged["score"] = merged["score"] + merged["score_r"]
        merged = merged.drop(columns=["%s_r" % col for col in both_cols])
        merged = add_ranks(merged)
        return merged

class ConcatenateTransformer(BinaryTransformerBase):
    name = "Concat"
    epsilon = 0.0001

    def __init__(self, operands, **kwargs):
        minimal_input = RANKED_DOCS
        minimal_output = RANKED_DOCS
        true_output = "input"
        super().__init__(operands=operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)

    def transform(self, topics_and_res):
        import pandas as pd
        # take the first set as the top of the ranking
        res1 = self.left.transform(topics_and_res)
        # identify the lowest score for each query
        last_scores = res1[['qid', 'score']].groupby('qid').min().rename(columns={"score" : "_lastscore"})

        # the right hand side will provide the rest of the ranking        
        res2 = self.right.transform(topics_and_res)

        
        intersection = pd.merge(res1[["qid", "docno"]], res2[["qid", "docno"]].reset_index())
        remainder = res2.drop(intersection["index"])

        # we will append documents from remainder to res1
        # but we need to offset the score from each remaining document based on the last score in res1
        # explanation: remainder["score"] - remainder["_firstscore"] - self.epsilon ensures that the
        # first document in remainder has a score of -epsilon; we then add the score of the last document
        # from res1
        first_scores = remainder[['qid', 'score']].groupby('qid').max().rename(columns={"score" : "_firstscore"})

        remainder = remainder.merge(last_scores, on=["qid"]).merge(first_scores, on=["qid"])
        remainder["score"] = remainder["score"] - remainder["_firstscore"] + remainder["_lastscore"] - self.epsilon
        remainder = remainder.drop(columns=["_lastscore",  "_firstscore"])

        # now bring together and re-sort
        # this sort should match trec_eval
        rtr = pd.concat([res1, remainder]).sort_values(by=["qid", "score", "docno"], ascending=[True, False, True]) 

        # recompute the ranks
        rtr = add_ranks(rtr)
        return rtr

class ScalarProductTransformer(BinaryTransformerBase):
    """
        Multiplies the retrieval score by a scalar
    """
    arity = Arity.binary
    name = "ScalarProd"

    def __init__(self, operands, **kwargs):
        minimal_input = RANKED_DOCS
        minimal_output = RANKED_DOCS
        true_output = "input"
        super().__init__(operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_output, true_output=true_output)
        self.transformer = operands[0]
        self.scalar = operands[1]

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        res["score"] = self.scalar * res["score"]
        return res

class RankCutoffTransformer(BinaryTransformerBase):
    """
        Applies a rank cutoff for each query
    """
    arity = Arity.binary
    name = "RankCutoff"

    def __init__(self, operands, **kwargs):
        operands = [operands[0], Scalar(str(operands[1]), operands[1])] if isinstance(operands[1], int) else operands
        minimal_input = RANKED_DOCS
        minimal_outout = RANKED_DOCS
        true_output = "input"
        super().__init__(operands, **kwargs, minimal_input=minimal_input, minimal_output=minimal_outout, true_output=true_output)
        self.transformer = operands[0]
        self.cutoff = operands[1]
        if self.cutoff.value % 10 == 9:
            warn("Rank cutoff off-by-one bug #66 now fixed, but you used a cutoff ending in 9. Please check your cutoff value. ", DeprecationWarning, 2)

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        if not "rank" in res.columns:
            assert False, "require rank to be present in the result set"

        # this assumes that the minimum rank cutoff is model.FIRST_RANK, i.e. 0
        res = res[res["rank"] < self.cutoff.value]
        return res
    
class ApplyTransformerBase(TransformerBase):
    """
        A base class for Apply*Transformers
    """
    def __init__(self, fn, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.verbose = verbose

class ApplyForEachQuery(ApplyTransformerBase):
    def __init__(self, fn,  *args, add_ranks=True, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score 
        """
        super().__init__(fn, *args, **kwargs, family=RERANKING, true_output='input')
        self.add_ranks = add_ranks
    
    def transform(self, res):
        rtr = pd.concat(self.fn(group) for qid, group in res.groupby("qid"))
        if self.add_ranks:
            rtr = add_ranks(rtr)
        return rtr


class ApplyDocumentScoringTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform document scoring. The supplied function 
        should take as input one row, and return a float for the score of the document.
        
        Usually accessed using pt.apply.doc_score()::

            def _score_fn(row):
                return float(row["url".count("/")])
            
            pipe = pt.BatchRetrieve(index) >> pt.apply.doc_score(_score_fn)
    """
    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score 
        """
        super().__init__(fn, *args, **kwargs, family=RERANKING, true_output='input')
    
    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            tqdm.pandas(desc="pt.apply.doc_score", unit="d")
            outputRes["score"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["score"] = outputRes.apply(fn, axis=1)
        outputRes = add_ranks(outputRes)
        return outputRes

class ApplyDocFeatureTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform feature scoring. The supplied function 
        should take as input one row, and return a numpy array for the features of the document.
        
        Usually accessed using pt.apply.doc_features()::

            def _feature_fn(row):
                return numpy.array([len(row["url"], row["url".count("/")])
            
            pipe = pt.BatchRetrieve(index) >> pt.apply.doc_features(_feature_fn) >> pt.LTRpipeline(xgBoost())
    """
    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns a new numpy array representing the features of that document
        """
        super().__init__(fn, *args, **kwargs, family=FEATURE_SCORING, true_output=lambda x: list(x) + ['features'])

    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            tqdm.pandas(desc="pt.apply.doc_features", unit="d")
            outputRes["features"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["features"] = outputRes.apply(fn, axis=1)
        return outputRes

class ApplyQueryTransformer(ApplyTransformerBase):
    """
        Implements a query rewriting transformer by passing a function to perform the rewriting. The function should take
        as input one row, and return the string form of the new query.
        
        Usually accessed using pt.apply.query() passing it the function::

            def _rewriting_fn(row):
                return row["query"] + " extra words"
            
            pipe = pt.apply.query(_rewriting_fn) >> pt.BatchRetrieve(index)

        Similarly, a lambda function can also be used::

            pipe = pt.apply.query(lambda row: row["query"] + " extra words") >> pt.BatchRetrieve(index)

    """
    def __init__(self, fn, *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing a query, and returns the new string query 
             - verbose (bool): Display a tqdm progress bar for this transformer
        """
        super().__init__(fn, *args, **kwargs, family=QUERY_REWRITE, true_output='input')

    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            tqdm.pandas(desc="pt.apply.query", unit="d")
            outputRes["query"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["query"] = outputRes.apply(fn, axis=1)
        return outputRes

class ApplyGenericTransformer(ApplyTransformerBase):
    """
    Allows arbitrary pipelines components to be written as functions. The function should take as input
    a dataframe, and return a new dataframe. The function should abide by the main contracual obligations,
    e.g. updating then "rank" column.

    This class is normally accessed through pt.apply.generic()

    If you are scoring, query rewriting or calculating features, it is advised to use one of the other
    variants.

    It is up to the creator of the generic transformer to pass the necessary keyword arguments in order to allow for
    pipeline validation
    
    Example::
        
        # this pipeline would remove all but the first two documents from a result set
        lp = ApplyGenericTransformer(lambda res : res[res["rank"] < 2])

        pipe = pt.BatchRetrieve(index) >> lp

    """

    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda DataFrame, and returns a new Pandas DataFrame 
        """
        super().__init__(fn, *args, **kwargs)

    def transform(self, inputRes):
        fn = self.fn
        return fn(inputRes)

@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.ApplyGenericTransformer")
class LambdaPipeline(ApplyGenericTransformer):
    pass

class FeatureUnionPipeline(NAryTransformerBase):
    """
        Implements the feature union operator.

        Example::
            cands = pt.BatchRetrieve(index wmodel="BM25")
            pl2f = pt.BatchRetrieve(index wmodel="PL2F")
            bm25f = pt.BatchRetrieve(index wmodel="BM25F")
            pipe = cands >> (pl2f ** bm25f)
    """
    name = "FUnion"

    def __init__(self, operands, **kwargs):
        minimal_input = DOCS
        super().__init__(operands=operands, **kwargs, minimal_input=minimal_input)

    def transform(self, inputRes):
        if not "docno" in inputRes.columns and "docid" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker, but input did not have either docno or docid columns, found columns were %s" %  str(inputRes.columns))

        num_results = len(inputRes)
        import numpy as np

        # a parent could be a feature union, but it still passes the inputRes directly, so inputRes should never have a features column
        if "features" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker. They can be nested, but input should not contain a features column; found columns were %s" %  str(inputRes.columns))
        
        all_results = []

        for i, m in enumerate(self.models):
            #IMPORTANT this .copy() is important, in case an operand transformer changes inputRes
            results = m.transform(inputRes.copy())
            if len(results) == 0:
                raise ValueError("Got no results from %s, expected %d" % (repr(m), num_results) )
            assert not "features_x" in results.columns 
            assert not "features_y" in results.columns
            all_results.append( results )

    
        for i, (m, res) in enumerate(zip(self.models, all_results)):
            #IMPORTANT: dont do this BEFORE calling subsequent feature unions
            if not "features" in res.columns:
                if not "score" in res.columns:
                    raise ValueError("Results from %s did not include either score or features columns, found columns were %s" % (repr(m), str(res.columns)) )

                if len(res) != num_results:
                    warn("Got number of results different expected from %s, expected %d received %d, feature scores for any missing documents be 0, extraneous documents will be removed" % (repr(m), num_results, len(res)))
                    all_results[i] = res = inputRes[["qid", "docno"]].merge(res, on=["qid", "docno"], how="left")
                    res["score"] = res["score"].fillna(value=0)

                res["features"] = res.apply(lambda row : np.array([row["score"]]), axis=1)
                res.drop(columns=["score"], inplace=True)
            assert "features" in res.columns
            #print("%d got %d features from operand %d" % ( id(self) ,   len(results.iloc[0]["features"]), i))

        def _concat_features(row):
            assert isinstance(row["features_x"], np.ndarray)
            assert isinstance(row["features_y"], np.ndarray)
            
            left_features = row["features_x"]
            right_features = row["features_y"]
            return np.concatenate((left_features, right_features))
        
        def _reduce_fn(left, right):
            import pandas as pd
            both_cols = set(left.columns) & set(right.columns)
            both_cols.remove("qid")
            both_cols.remove("docno")
            both_cols.remove("features")
            rtr = pd.merge(left, right, on=["qid", "docno"])            
            rtr["features"] = rtr.apply(_concat_features, axis=1)
            rtr.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
            rtr.drop(columns=["features_x", "features_y"] + ["%s_y" % col for col in both_cols], inplace=True)
            return rtr
        
        from functools import reduce
        final_DF = reduce(_reduce_fn, all_results)

        # final_DF should have the features column
        assert "features" in final_DF.columns

        # we used .copy() earlier, inputRes should still have no features column
        assert not "features" in inputRes.columns

        # final merge - this brings us the score attribute from any previous transformer
        both_cols = set(inputRes.columns) & set(final_DF.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        final_DF = inputRes.merge(final_DF, on=["qid", "docno"])
        final_DF.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
        final_DF.drop(columns=["%s_y" % col for col in both_cols], inplace=True)
        # remove the duplicated columns
        #final_DF = final_DF.loc[:,~final_DF.columns.duplicated()]
        assert not "features_x" in final_DF.columns 
        assert not "features_y" in final_DF.columns 
        return final_DF

class ComposedPipeline(NAryTransformerBase):
    """ 
        This class allows pipeline components to be chained together using the "then" operator.

        :Example:

        >>> comp = ComposedPipeline([ DPH_br, ApplyGenericTransformer(lambda res : res[res["rank"] < 2])])
        >>> # OR
        >>> # we can even use lambdas as transformers
        >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
        >>> # this is equivelent
        >>> # comp = DPH_br >> lambda res : res[res["rank"] < 2]]
    """
    name = "Compose"

    def transform(self, topics):
        for m in self.models:
            topics = m.transform(topics)
        return topics

    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va=None, qrels_va=None):
        """
        This is a default implementation for fitting a pipeline. The assumption is that
        all EstimatorBase be composed with a TransformerBase. It will execute any pre-requisite
        transformers BEFORE executing the fitting the stage.
        """
        for m in self.models:
            if isinstance(m, EstimatorBase):
                m.fit(topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va)
            else:
                topics_or_res_tr = m.transform(topics_or_res_tr)
                # validation is optional for some learners
                if topics_or_res_va is not None:
                    topics_or_res_va = m.transform(topics_or_res_va)
