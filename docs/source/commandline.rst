
.. _l-command-line-doc:

============
Command Line
============

.. index:: command line, help

:epkg:`ML.net` has a command line available through the following
syntax:

::

    dotnet ./bin/AnyCPU.Debug/Microsoft.ML.Console/netcoreapp2.1/MML.dll <string>
    
The ``<string>`` is either ``?`` to get help or a command.

.. contents::
    :local:
    
List of available commands
==========================

.. mlcmd::
    :toggle: out
    :showcode:
    :current:
    
    ?


The whole list of commands documented at
:ref:`l-commands`.
    
Help on a particulier command
=============================


.. mlcmd::
    :toggle: out
    :showcode:
    :current:
    
    ? chain

List of available object kind
=============================

.. mlcmd::
    :toggle: out
    :showcode:
    :current:
    
    ? kind=

List of available trainers
==========================

.. mlcmd::
    :toggle: out
    :showcode:
    :current:
    
    ? kind=trainer

Help on a particular object
===========================

.. mlcmd::
    :toggle: out
    :showcode:
    :current:
    
    ? lr

.. _l-cmd-multi-class-logistic-regression:

One example: train a logistic regression
========================================


.. mlcmd::
    :toggle: out
    :showcode:
    :current:

    train
    data=iris.txt
    loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+}
    tr=mlr{maxiter=5}
    out=logistic_regression.zip

Chaining a training with an export
==================================

.. index:: chain

The following command line trains a model then
exports it to :epkg:`ONNX`
(see also :ref:`l-onnx`).

.. mlcmd::
    :toggle: out
    :showcode:
    :current:

    chain

    cmd = train{
        data=iris.txt
        loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+}
        tr=mlr{maxiter=5}
        out=logistic_regression.zip
    }

    cmd = saveonnx{
        in = logistic_regression.zip
        onnx = logistic_regression.onnx
        domain = ai.onnx.ml
        idrop = Label
    }

Produces a C# code to predict
=============================

.. mlcmd::
    :toggle: out
    :showcode:
    :current:

    chain

    cmd = train{
        data=iris.txt
        loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+}
        tr=mlr{maxiter=5}
        out=logistic_regression.zip
    }

    cmd = codegen{
        in = logistic_regression.zip
        cs = logistic_regression.cs
    }

The second command produces a :epkg:`C#` which can be used
to compute predictions with a C# implementation.

.. runpython::
    :current:
    
    with open("logistic_regression.cs", "r", encoding="utf-8") as f:
        print(f.read())
