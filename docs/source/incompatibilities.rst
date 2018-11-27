

====================
ML.net customization
====================

This repository includes :epkg:`ML.net` as a submodule,
it does not directly points to the main repository but
to modified version of it which will be eventually merged.
The submodule points to branch *ext* from
`xadupre/machinelearning <https://github.com/xadupre/machinelearning/tree/ext>`_.
Many changes changes were introduced and the custom extensions probably 
would be compile against the current nuget package 
`Microoft.ML <https://www.nuget.org/packages/Microsoft.ML/>`_
without a significant amount of work.
This will wait until :epkg:`ML.net`'s API stabilizes.

.. contents::
    :local:

Console.BufferWidth
===================

Commit 
`catch exception due to Console.BufferWidth <https://github.com/xadupre/machinelearning/commit/7b891369a23bb3955972cee515ce2a7753bcae68>`_.
`Console.BufferWidth <https://docs.microsoft.com/fr-fr/dotnet/api/system.console.bufferwidth>`_ is not always
available when the command line is called from an external program.
The commit catches any exception raised due to that.

OVA
===

Commit
`improves ova <https://github.com/xadupre/machinelearning/commit/5fc9981dae162975bba0dfae20f3d8c3eb00d821>`_.
It does not fix the following issue
`#1259 <https://github.com/dotnet/machinelearning/issues/1259>`_ but could be part of it.

Warning as errors
=================

The compilation failed due to a couple of warnings treated as error
on appveyor and Visual Studio 2015. The option was removed:
`Remove option /WX for native libraries <https://github.com/xadupre/machinelearning/commit/a7eb9efb54a0849bb76279a807ab4fef7b8752d2>`_.

Internal
========

A couple of classes where duplicated because they became internal and
then many internal where turned into public due to
[BestFriendAttribute](https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Core/BestFriendAttribute.cs).

