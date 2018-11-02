
.. index:: command line

============
Command Line
============

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
