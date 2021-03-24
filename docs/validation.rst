.. _pt.validation:

Pipeline Validation
-------------------

PyTerrier tranformers are plugged together using operators to form pipelines. However, it is possible to construct pipelines that represent invalid operations. Pipeline validation allows constructed pipelines to be checked for correctness before running any expensive retrieval operations. 

Tranformers are equipped with a `validate()` method, which takes as input the dataframe that we are validating the pipeline against.
This method will check that a given transformer receives the required input dataframe. 
For operator transformers, the validate method traverses the the pipeline and checks that output dataframes from operand transformers are suitable for given operation.

Calling the method validate will return the columns of dataframe that is output when that transformer transforms the given input.
i.e. For a dataframe `input`, calling `t.validate(input)` will return the same set of column names as calling `t.transform(input).columns`.

If validation discovers that there is a mismatch between the input to a tranformer and what it needs, then it raises a `PipelineError`.

Transformer Families
====================

To assist with pipeline validation, we utilise the idea of transformer families. 
These are the categorisations of transformers, based on what dataframes they need as input, and what dataframes they give as output.  

Constants which represent these transformer families and the common dataframe types are accesible within `pt.transformer.Family` and `pt.model` respectively. 

+------------------+-----------------------------------------+----------------------+-------------------------------+
+ Family           | PyTerrier constant                      | Input                | Output                        |
+==================+=========================================+======================+===============================+
| Query rewriting  | `pt.transformer.Family.QUERY_REWRITE`   | pt.model.QUERIES     | pt.model.QUERIES              |
+------------------+-----------------------------------------+----------------------+-------------------------------+
| Retrieval        | `pt.transformer.Family.RETRIEVAL`       | pt.model.QUERIES     | pt.model.RANKED_DOCS          |
+------------------+-----------------------------------------+----------------------+-------------------------------+
| Query expansion  | `pt.transformer.Family.QUERY_EXPANSION` | pt.model.RANKED_DOCS | pt.model.QUERIES              |
+------------------+-----------------------------------------+----------------------+-------------------------------+
| Re-ranking       | `pt.transformer.Family.RERANKING`       | pt.model.RANKED_DOCS | pt.model.RANKED_DOCS          |
+------------------+-----------------------------------------+----------------------+-------------------------------+
| Feature scoring  | `pt.transformer.Family.FEATURE_SCORING` | pt.model.QUERIES     | pt.model.RANKED_DOCS_FEATURES |
+------------------+-----------------------------------------+----------------------+-------------------------------+

Making Your Transformers Suitable For Validation
================================================

Transformers have the attributes `minimal_input` and `minimal_output`. These correspond to the input and output of transformer families. 

When you create a custom transformer, these attributes need to be defined for validation. 
`minimal_input` will be the list of column names that the transformer requires, needed to validate the input to a transformer, and `minimal_output` will be the list of column names that the transformer will return, needed for pipelines where the output of one transformer becomes the input to another.

The simplest way to define these attributes when creating a transformer is using the keyword argumen `family`. This will use the built in mapping between a transformer family and its input and output::

  # transformer1.minimal_input = ['qid', 'query']
  # transformer1.minimal_output = ['qid', 'query', 'docno', 'score', 'rank']
  transformer1 = pt.apply.generic(fn, family=pt.transformer.Family.RETRIEVAL)

If a transformer does not fit into an existing family, it is possible to define the input and output manually upon creation.
This can be a list of column names, or a function that returns a list of columns by manipulating some input dataframe::

  # we can define attributes manually
  transformer2 = pt.apply.generic(fn, minimal_input=pt.model.QUERIES, minimal_output=pt.model.QUERIES + ['url'])
  
Internally, there is another attribute transformers posses, `true_output`. This is the true representation of what columns a transformer will return.
By default, this is the same as `minimal_output`. 
However, this can be defined seperately if there is some divergance between the minimal output and true output. 

The most common use case is when a transformer passes all input columns through as output. In this case we can set `true_output="input"`.
Alternatively, it can also be a list of columns or a function that returns a list of columns, just like `minimal_output`::

  # transformer3 belongs to query rewriting family, but will return the same dataframe columns that it gets as input
  transformer3 = pt.apply.generic(fn, family=pt.transformer.Family.QUERY_REWRITE, trueoutput="input")


