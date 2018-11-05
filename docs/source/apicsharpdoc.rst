
==========
CSharp API
==========

:epkg:`ML.net` is a :epkg:`C#` library.
This page compiles a couple of examples
and some exploration.

.. contents::
    :local:



Example with Iris DataSet in C#
===============================


.. runcsharpml::
    :entrypoint: example_iris
    :showcode:
    :language: C#

    public class IrisObservation
    {
        [Column("0")]
        [ColumnName("Label")]
        public string Label;

        [Column("1")]
        public float Sepal_length;

        [Column("2")]
        public float Sepal_width;

        [Column("3")]
        public float Petal_length;

        [Column("4")]
        public float Petal_width;
    }
    
    public class IrisPrediction
    {
        public uint PredictedLabel;

        [VectorType(4)]
        public float[] Score;
    }
    
    public static void example_iris()
    {
        var iris = string.Format("{0}/iris.txt", RELPATH);
        
        using (var env = new ConsoleEnvironment())
        {
            var args = new TextLoader.Arguments()
                {
                    Separator = "\t",
                    HasHeader = true,
                    Column = new[] {
                        new TextLoader.Column("Label", DataKind.R4, 0),
                        new TextLoader.Column("Sepal_length", DataKind.R4, 1),
                        new TextLoader.Column("Sepal_width", DataKind.R4, 2),
                        new TextLoader.Column("Petal_length", DataKind.R4, 3),
                        new TextLoader.Column("Petal_width", DataKind.R4, 4),
                    }
                };

            var reader = new TextLoader(env, args);
            var concat = new ColumnConcatenatingEstimator(env,
                                "Features", "Sepal_length",
                                "Sepal_width", "Petal_length", "Petal_width");
            var km = new KMeansPlusPlusTrainer(env, "Features", clustersCount: 3);
            var pipeline = concat.Append(km);

            IDataView trainingDataView = reader.Read(new MultiFileSource(iris));
            var model = pipeline.Fit(trainingDataView);

            var obs = new IrisObservation()
            {
                Sepal_length = 3.3f,
                Sepal_width = 1.6f,
                Petal_length = 0.2f,
                Petal_width = 5.1f,
            };

            var engine = model.MakePredictionFunction<IrisObservation, 
                                                      IrisPrediction>(env);
            var res = engine.Predict(obs);
            Console.WriteLine("------------");
            Console.WriteLine("PredictedLabel: {0}", res.PredictedLabel);
            Console.WriteLine("Score: {0}", string.Join(", ", 
                                res.Score.Select(c => c.ToString())));
        }
    }


DataFrame in C#
===============

This code can be shortened with the use of DataFrame
and a custom implemantation of the pipeline.
It is a mix between the command line and the :epkg:`C#`.


.. runcsharpml::
    :entrypoint: dataframe_iris
    :showcode:
    :language: C#
    
    public static void dataframe_iris()
    {
        var iris = string.Format("{0}/iris.txt", RELPATH);
        
        using (var env = new ConsoleEnvironment())
        {
            var df = DataFrameIO.ReadCsv(iris, sep: '\t', 
                        dtypes: new ColumnType[] { NumberType.R4 });
            var concat = string.Format("Concat{{col=Features:{0},{1}}}", 
                                       df.Columns[1], df.Columns[2]);
            var pipe = new ScikitPipeline(new[] { concat }, "mlr");
            pipe.Train(df, "Features", "Label");
            
            DataFrame pred = null;
            pipe.Predict(df, ref pred);
            Console.WriteLine(pred.Head());
        }
    }
