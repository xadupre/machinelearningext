# About machine learning

This project proposes some extension to
[machinelearning](https://github.com/dotnet/machinelearning)
written in C#.
Work in progress.

[![TravisCI](https://travis-ci.org/xadupre/machinelearningext.svg?branch=master)](https://travis-ci.org/xadupre/machinelearningext)
[![Build status](https://ci.appveyor.com/api/projects/status/cb0xos4p3xe1bqmg?svg=true)](https://ci.appveyor.com/project/xadupre/machinelearningext)
[![CircleCI](https://circleci.com/gh/xadupre/machinelearningext.svg?style=svg)](https://circleci.com/gh/xadupre/machinelearningext)

## Build

On windows: ``build.cmd``.

On Linux: ``build.sh``.

## Documentation

``doxygen conf.dox``

### Example 1: inner API

This example relies on the inner API, mostly used
inside components of ML.net.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrame.ReadCsv(iris, sep: '\t',
                           dtypes: new DataKind?[] { DataKind.R4 });
                           
// We add a transform to concatenate two features in one vector columns.
var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);

// We create training data by mapping roles to columns.
var trainingData = env.CreateExamples(conc, "Feature", label: "Label");

// We create a trainer, here a One Versus Rest with a logistic regression as inner model.
var trainer = env.CreateTrainer("ova{p=lr}");

using (var ch = env.Start("test"))
{
    // We train the model.
    var pred = trainer.Train(env, ch, trainingData);
    
    // We compute the prediction (here with the same training data but it should not be the same).
    var scorer = trainer.GetScorer(pred, trainingData, env, null);
    
    // We store the predictions on a file.
    DataFrame.ViewToCsv(env, scorer, "iris_predictions.txt");
    
    // Or we could put the predictions into a dataframe.
    var predictions = DataFrame.ReadView(scorer);
    
    // And access one value...
    var v = predictions.iloc[0, 7];
    Console.WriteLine("PredictedLabel: {0}", v);
}
```

The current interface of 
[DataFrame](https://github.com/xadupre/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs)
is not rich. It will improve in the future.

### Example 2: common API

This is the same example based on
[Iris Classification](https://github.com/dotnet/machinelearning-samples/tree/master/samples/getting-started/MulticlassClassification_Iris)
but using the new class DataFrame. It is not necessary anymore
to create a class specific to the data used to train. It is a
little bit less efficient.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrame.ReadCsv(iris, sep: '\t',
                           dtypes: new DataKind?[] { DataKind.R4 });

var importData = df.EPTextLoader(iris, sep: '\t', header: true);
var learningPipeline = new LearningPipeline();
learningPipeline.Add(importData);
learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
var predictor = learningPipeline.Train();
var predictions = predictor.Predict(df);

// We store the predictions on a file.
DataFrame.ViewToCsv(env, scorer, "iris_predictions.txt");

// Or we could put the predictions into a dataframe.
var df = DataFrame.ReadView(predictions);

// And access one value...
var v = df.iloc[0, 7];
Console.WriteLine("{0}: {1}", vdf.Schema.GetColumnName(7), v.iloc[0, 7]);
```
