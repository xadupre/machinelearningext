

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


Warning as errors
=================

The compilation failed due to a couple of warnings treated as error
on appveyor and Visual Studio 2015. The option was removed:
`Remove option /WX for native libraries <https://github.com/xadupre/machinelearning/commit/a7eb9efb54a0849bb76279a807ab4fef7b8752d2>`_.

Internal
========

A couple of classes where duplicated because they became internal and
then many internal where turned into public due to
`BestFriendAttribute <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Core/BestFriendAttribute.cs>`_.
See also `Switch internal to public to allow external contributions <https://github.com/xadupre/machinelearning/commit/7cc342326dd0c24504dbf23078d129c3f5da09a0>`_.

Other changes

Empty cursor
============

The following commit
`Avoids splitting a cursor if the set it walks through is empty or very small <https://github.com/xadupre/machinelearning/commit/b137aa9f0a131d92b440d28c9c280c0be3cb2cc2>`_
avoids splitting a single cursor in multiple thread if the cursor
contains only one row even if requested.
This is needed by the InfiniteLoopView cursor which outputs
only one row each time.

