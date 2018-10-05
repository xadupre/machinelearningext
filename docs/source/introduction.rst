
============
Introduction
============

:epkg:`ML.net` is machine learning library
implement in :epkg:`C#` by :epkg:`Microsoft`.
This projet aims at showing how to extend it with
custom tranforms or learners.
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

.. index:: Inner API, streaming API

Inner API
=========

Basically designed as a streaming API.

