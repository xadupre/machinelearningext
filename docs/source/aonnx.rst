
.. _l-onnx:

===============
ML.net and ONNX
===============

:epkg:`ONNX` format provides a way to describe
a machine learned model. The main purpose is
to deploy model into production in such a way
that it is optimized to compute predictions.

.. contents::
    :local:

About ONNX
==========

Every machine learned model can be described as 
a sequence of basic numerical operations:
``+``, ``*``, ... Let's see for example what
it looks like for a :epkg:`linear regression`.
Let's first train a model:

.. runpython::
    :showcode:
    :store:
    :warningout: ImportWarning

    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    diabetes_X_train = diabetes.data[:-20]
    diabetes_X_test  = diabetes.data[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test  = diabetes.target[-20:]
    
    from sklearn.linear_model import LinearRegression
    clr = LinearRegression()
    clr.fit(diabetes_X_train, diabetes_y_train)
    print(clr)
    
The model is trained and we can display the coefficients.

.. runpython::
    :showcode:
    :restore:
    :store:
    
    print(clr.coef_)
    print(clr.intercept_)

The model can be deployed as is with the module :epkg:`scikit-learn`.
It is simple but slow, more than 10 times than a pure :epkg:`python`
implementation for this particular example.

.. runpython::
    :showcode:
    :restore:
    :store:
    
    from textwrap import wrap
    code = str(clr.intercept_) + " + " + " + ".join("x[{0}]*({1})".format(i, c) for i, c in enumerate(clr.coef_))
    print("\n".join(wrap(code)))

Next figure explores various rewriting of this linear models,
including :epkg:`C++` ones with :epkg:`numba` or :epkg:`cffi`
and :epkg:`AVX` instructions.

.. image:: http://www.xavierdupre.fr/app/jupytalk/helpsphinx/_images/onnx_deploy_112_0.png
    :target: http://www.xavierdupre.fr/app/jupytalk/helpsphinx/notebooks/onnx_deploy.html

This solution is still tied to :epkg:`Python` even though it reduces
the number of dependencies. It is one option some 
followed when that was really needed. Linear models are easy,
decision trees, random forests a little bit less, deep learning
models even less. It is now a common need
and that what be worth having a common solution.

That's where :epkg:`ONNX` takes place. It provides a common way to describe 
machine learning models with high level functions specialied for
machine learning: :epkg:`onnx ml functions`.


ONNX description of a linear model
==================================

Module :epkg:`onnxmltools` implements a subset of machine
learned models for :epkg:`scikit-learn` or :epkg:`lightgbm`.
The conversion requires the user to give a name and
the input shape.

.. runpython::
    :showcode:
    :restore:
    :warningout: ImportWarning

    from onnxmltools import convert_sklearn
    from onnxmltools.utils import save_model
    from onnxmltools.convert.common.data_types import FloatTensorType
    
    onnx_model = convert_sklearn(clr, 'linear regression',
                                 [('input', FloatTensorType([1, 10]))])
    save_model(onnx_model, 'lr_diabete.onnx')


Let's see what the :epkg:`ONNX` format looks
like by using module :epkg:`onnx`.

.. runpython::
    :showcode:
    :toggle: out
    
    import onnx
    model = onnx.load('lr_diabete.onnx')
    print(model)

The result shows one main function which is a linear
regression. Every coefficient is converted by default into
floats. :epkg:`ONNX` assumes every machine learned models
can be described by a set of these functions or more
precisely a pipeline. It also describes the input and output.

    
ONNX conversion with ML.net
===========================

:epkg:`ML.net` is a machine learning library written
in :epkg:`C#`. It implements many learners
(see :ref:`l-ml-net-components`) which can be run
from :epkg:`C#` or from the command line.
Let's first split the dataset into train and test
then save it on disk.


.. runpython::
    :showcode:
    :current:
    :warningout: ImportWarning
    
    from sklearn.datasets import load_diabetes
    from pandas import DataFrame
    from sklearn.model_selection import train_test_split
    
    diabetes = load_diabetes()
    df = DataFrame(diabetes.data, columns=["F%d" % i for i in range(diabetes.data.shape[1])])
    df["Label"] = diabetes.target
    df_train, df_test = train_test_split(df)
    df_train.to_csv("diabetes_train.csv", index=False)
    df_test.to_csv("diabetes_test.csv", index=False)

The following command line trains a model,
evaluates it on the test set, saves it as a zip
format and finally convert it into :epkg:`ONNX` format.


.. mlcmd::
    :toggle: out
    :showcode:
    :current:

    chain
    
    cmd=traintest{
        data=diabetes_train.csv
        test=diabetes_test.csv
        loader=text{col=Label:R4:10 col=Features:R4:0-9 header=+ sep=,}
        tr=ols
        out=lr_diabete_cs.zip
    }
    
    cmd=saveonnx{
        in=lr_diabete_cs.zip
        onnx=lr_diabete_cs.onnx
        domain=ai.onnx.ml
        idrop=Label
        odrop=Features1
    }

Let's display the outcome.
Parameters *idrop* and *odrop*
defines which input and output are not necessary.

.. runpython::
    :showcode:
    :toggle: out
    :current:
    
    import onnx
    model = onnx.load('lr_diabete_cs.onnx')
    print(model)

Two different machine learning libraries produce
a similar model finally described the same way.
The second one includes a :ref:`l-scaler-transform`.

ONNX serialization
==================

:epkg:`ONNX` internally relies on :epkg`Google Protobuf`
which is used here as an efficient way to serialize the data.
The outcome is compact and optimized for a fast access.


ONNX runtime
============

Once the model is described with a common language,
it becomes possible to separate training and testing.
The training still happens with a standard machine library,
the predictions are computed on a different machine with
a dedicated runtime. :epkg:`onnxruntime` is one of them
which has a :epkg:`python` interface.
The following example prints the inputs and outputs
and then compute the predictions for one random example.

.. runpython::
    :showcode:
    :current:
    
    import onnxruntime as rt
    import numpy
    from sklearn.datasets import load_diabetes
    
    sess = rt.InferenceSession("lr_diabete_cs.onnx")
    
    for i in sess.get_inputs():
        print('Input:', i)
    for o in sess.get_outputs():
        print('Output:', o)

    X = load_diabetes().data
    x = X[:1].astype(numpy.float32)
    res = sess.run(None, {'Features': x})
    for o, r in zip(sess.get_outputs(), res):
        print(o, "=", r)
    
The last result is the expected one.
The runtime does not depend on :epkg:`scikit-learn`
or :epkg:`ML.net` and runs on CPU or GPU.
It is implemented in :epkg:`C++` and is optimized for
deep learning.
