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

**Example 1: inner API**

This example relies on the inner API, mostly used
inside components of ML.net.

```CSharp
var env = new TlcEnvironment();
var iris = "iris.txt";

// We read the text data and create a dataframe / dataview.
var df = DataFrame.ReadCsv(env.Register("DataFrame"),
                           iris, sep: '\t',
                           dtypes: new DataKind?[] { DataKind.R4 });
                           
// We add a transform to concatenate two features in one vector columns.
var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);

// We create training data by mapping roles to columns.
var trainingData = env.CreateExamples(conc, "Feature", label: "Label");

// We create a trainer, here a One Versus Rest with a logistic regression as inner model.
string loadName;
var trainer = env.CreateTrainer("ova{p=lr}", out loadName);

using (var ch = env.Start("test"))
{
    // We train the model.
    var pred = TrainUtils.Train(env, ch, trainingData, trainer, loadName, null, null, 0, false);
    
    // We compute the prediction (here with the same training data but it should not be the same).
    var scorer = ScoreUtils.GetScorer(pred, trainingData, env, null);
    
    // We store the predictions on a file.
    DataFrame.ViewToCsv(env, scorer, "iris_predictions.txt");
    
    // Or we could put the predictions into a dataframe.
    var predictions = DataFrame.ReadView(env.Register("predictions"), scorer);
    
    // And access one value...
    var v = predictions.iloc[0, 7];
    Console.WriteLine("PredictedLabel: {0}", v);
}
```

The current interface of 
[DataFrame](https://github.com/xadupre/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs)
is not rich. It will improve in the future.

**Example 2: common API**

This example relies on the inner API, mostly used
inside components of ML.net.

```CSharp
```
