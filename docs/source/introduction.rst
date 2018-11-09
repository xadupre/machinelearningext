
============
Introduction
============

:epkg:`ML.net` is a machine learning library
implemented in :epkg:`C#` by :epkg:`Microsoft`.
This projet aims at showing how to extend it with
custom tranforms or learners.
It implements standard abstraction in C# such as
dataframes and pipeline following the 
:epkg:`scikit-learn` API. :epkg:`ML.net` implements
two API. The first one structured as a streaming API
merges every experiment in a single sequence of transform
and learners possibly handling one out-of-memory dataset.
The second API is built on the top of the first one
and proposes an easier way to build pipeline
with multiple datasets. This second API is also used
by wrapper to other language such as :epkg:`NimbusML`.
Let's see first how this library can be used
without any addition.

.. contents::
    :local:

.. index:: command line

Command line
============

:epkg:`ML.net` proposes some sort of simple language
to define a simple machine learning pipeline.
We use it on :epkg:`Iris` data to 
train a logistic regression.

.. literalinclude:: iris.txt
    :lines: 1-6

The pipeline is simply define by a logistic
regression named ``mlr`` for
``MultiLogisticRegression``.
Options are defined inside ``{...}``.
The parameter *data=* specifies the data file,
*loader=* specifies the format and column names.

.. index:: help

.. mlcmd::
    :toggle: out
    :showcode:
    :current:

    train
    data=iris.txt
    loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+}
    tr=mlr{maxiter=5}
    out=logistic_regression.zip

The documentation of every component is available
through the command line. An exemple for
:ref:`l-multi-class-logistic-regression`:

.. mlcmd::
    :showcode:

    ? mlr

More example can be found at :ref:`l-command-line-doc`.
The command line is usually the preferred way to use 
the library. It does not requires a huge setup and
and makes the training easier. Online predictions
require :epkg:`C#` but command 
:ref:`l-generate-sample-prediction-code` may help
in that regard.

.. index:: Inner API, streaming API

New components in this extension
================================

.. list-table::
    :widths: 10 10 3 3
    :header-rows: 1
   
    * - Name
      - Kind
      - Streaming API
      - Pipeline API
    * - :ref:`l-describe-transform`
      - :ref:`l-transforms-all`
      - X
      -
    * - :ref:`l-detrendtransform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-multitobinary`
      - :ref:`l-transforms-all`
      - X
      -
    * - :ref:`l-multitoranker`
      - :ref:`l-transforms-all`
      - X
      -
    * - :ref:`l-nearestneighborsmcc`
      - :ref:`l-multiclass-classification`
      - X
      - X
    * - :ref:`l-nearest-neighbors-transform`
      - :ref:`l-transforms-all`
      - X
      - X
    * - :ref:`l-nearestneighborsbc`
      - :ref:`l-binary-classification`
      - X
      - X
    * - :ref:`l-optics-ordering-transform`
      - :ref:`l-transforms-all`
      - X
      - X
    * - :ref:`l-optimized-one-vs-all`
      - :ref:`l-multiclass-classification`
      - X
      - 
    * - :ref:`l-opticstransform`
      - :ref:`l-transforms-all`
      - X
      - X
    * - :ref:`l-pass-through-transform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-polynomial-transform`
      - :ref:`l-transforms-all`
      - X
      - X
    * - :ref:`l-resample-transform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-run-prediction-for-a-transform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-scaler-transform`
      - :ref:`l-transforms-all`
      - X
      - X
    * - :ref:`l-select-tagged-view`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-sort-in-dataframe-transform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-split-train-test-transform`
      - :ref:`l-transforms-all`
      - X
      - 
    * - :ref:`l-train-and-tag-and-score-a-predictor`
      - :ref:`l-transforms-all`
      - X
      -
    * - :ref:`l-ulabeltor4label-transform`
      - :ref:`l-transforms-all`
      - X
      - 
