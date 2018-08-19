

===============
DataFrame in C#
===============

.. contents::
    :local:



The class :epkg:`DataFrame` replicates some functionalities
datascientist are used to in others languages such as
:epkg:`Python` or :epkg:`R`. It is possible to do basic operations
on columns:

.. code-block:: CSharp

    var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
    var df = DataFrame.ReadStr(text);
    df["AA+BB"] = df["AA"] + df["BB"];
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB
    0,1,text,1
    1,1.1,text2,2.1

Or:

.. code-block:: CSharp

    df["AA2"] = df["AA"] + 10;
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    1,1.1,text2,2.1,11

The next instructions change one value
based on a condition.

.. code-block:: CSharp

    df.loc[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed";
    Console.WriteLine(df.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    1,1.1,changed,2.1,11

A specific set of columns or rows can be extracted:

.. code-block:: CSharp

    var view = df[df.ALL, new [] {"AA", "CC"}];
    Console.WriteLine(view.ToString());

.. code-block:: text

    AA,CC
    0,text
    1,changed

The dataframe also allows basic filtering:

.. code-block:: CSharp

    var view = df[df["AA"] == 0];
    Console.WriteLine(view.ToString());

.. code-block:: text

    AA,BB,CC,AA+BB,AA2
    0,1,text,1,10
    