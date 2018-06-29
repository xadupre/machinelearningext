=============
Some concepts
=============

.. contents::
    :local:


Elements of code
================

Definitions
+++++++++++

* **active column**: a transform can replace a column by another or just skip one,
  in that case, the older column may remain but will be *hidden* as opposed to *active*.
* **row**: data unit for every data set, every row is usually independent
  from the others but not necessarily,
  not independant if there is a group column.
* **cursor**: row iterator on a dataset
* **getter**: each cursor does not give a pointer on a row (the type of row is 
  unknown from `ML.net <https://github.com/dotnet/machinelearning>`_ 's point of view),
  but a cursor defines a way to access a specific column through a getter.
  A getter is a function which
  retrieve the value for a column for the current row.
* **schema**: a list of typed columns


IDataView
+++++++++

A data view must define a schema
and an access to them through the same API. By default, you should assume rows
of the data are not order and can be accessed through multiple threads at the same time.
`IDataView <https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.runtime.data.idataview?view=ml-dotnet>`_ API.
        
Other interfaces
++++++++++++++++

* `ICursor <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Core/Data/ICursor.cs>`_,
  see `ICursor Notes <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Core/Data/ICursor.md>`_
* `IDataTransform <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Data/Data/IDataLoader.cs#L91>`_
* `IRow <https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.Core/Data/IDataView.cs#L154>`_

.. image:: cursor.png


Subleties
=========

Trainable transform
+++++++++++++++++++

The training usually happens when an object is instantiated
in `ML.net <https://github.com/dotnet/machinelearning>`_ and the next
step in the pipeline requires this transform to be instantiated.
The transforms in this extension postpones the training step
until some cursors are created:
`GetRowCursor <https://github.com/xadupre/machinelearningext/blob/master/machinelearningext/FeaturesTransforms/ScalerTransform.cs#L206>`_.

