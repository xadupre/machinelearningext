﻿// See the LICENSE file in the project root for more information.

#pragma warning disable
using System;
using System.Linq;
using Newtonsoft.Json;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.CommandLine;
using Scikit.ML.PipelineHelper;

using NearestNeighborsMultiClassClassificationTrainer = Scikit.ML.NearestNeighbors.NearestNeighborsMultiClassClassificationTrainer;
using EntryPointNearestNeighborsMultiClass = Scikit.ML.EntryPoints.EntryPointNearestNeighborsMultiClass;
using NearestNeighborsMultiClass = Scikit.ML.EntryPoints.NearestNeighborsMultiClass;


[assembly: LoadableClass(typeof(void), typeof(EntryPointNearestNeighborsMultiClass), null,
    typeof(SignatureEntryPointModule), NearestNeighborsMultiClass.Name)]



namespace Scikit.ML.EntryPoints
{
    #region Definition

    public static partial class EntryPointNearestNeighborsMultiClass
    {
        [TlcModule.EntryPoint(
            Name = EntryPointsConstants.EntryPointPrefix + NearestNeighborsMultiClass.Name,
            Desc = NearestNeighborsMultiClassClassificationTrainer.Summary,
            UserName = NearestNeighborsMultiClass.Name,
            ShortName = NearestNeighborsMultiClassClassificationTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, NearestNeighborsMultiClassClassificationTrainer.ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Train" + NearestNeighborsMultiClass.Name);
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return EntryPointsHelper.Train<NearestNeighborsMultiClassClassificationTrainer.ArgumentsEntryPoint,
                                           CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new NearestNeighborsMultiClassClassificationTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsNearestNeighborsMultiClassHelper
    {
        public static NearestNeighborsMultiClass.Output Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsMultiClass input)
        {
            var output = new NearestNeighborsMultiClass.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsMultiClass input, NearestNeighborsMultiClass.Output output)
        {
            exp.AddSerialize(EntryPointsConstants.EntryPointPrefix + NearestNeighborsMultiClass.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

    /// <summary>
    /// k-Nearest Neighbors trainer for Multi-Class Classification
    /// </summary>
    public sealed partial class NearestNeighborsMultiClass : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInput, Microsoft.ML.ILearningPipelineItem
    {
        public const string Name = nameof(NearestNeighborsMultiClass);

        public NearestNeighborsMultiClass()
        {
        }

        public NearestNeighborsMultiClass(string featureColumn = null, string labelColumn = null)
        {
            if (featureColumn != null)
                FeatureColumn = featureColumn;
            if (labelColumn != null)
                LabelColumn = labelColumn;
        }

        /// <summary>
        /// The data to be used for training
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> TrainingData { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Column to use for features
        /// </summary>
        public string FeatureColumn { get; set; } = "Features";

        /// <summary>
        /// Normalize option for the feature column
        /// </summary>
        public NormalizeOption NormalizeFeatures { get; set; } = NormalizeOption.Auto;

        /// <summary>
        /// Whether learner should cache input training data
        /// </summary>
        public CachingOptions Caching { get; set; } = CachingOptions.Auto;

        /// <summary>
        /// Column to use for labels
        /// </summary>
        public string LabelColumn { get; set; } = "Label";

        /// <summary>
        /// Number of neighbors to consider.
        /// </summary>
        [JsonProperty("k")]
        public int K { get; set; } = 5;

        /// <summary>
        /// Weighting strategy for neighbors
        /// </summary>
        [JsonProperty("algo")]
        public NearestNeighborsAlgorithm Algo { get; set; } = NearestNeighborsAlgorithm.kdtree;

        /// <summary>
        /// Weighting strategy for neighbors
        /// </summary>
        [JsonProperty("weight")]
        public NearestNeighborsWeights Weight { get; set; } = NearestNeighborsWeights.uniform;

        /// <summary>
        /// Distnace to use
        /// </summary>
        [JsonProperty("distance")]
        public NearestNeighborsDistance Distance { get; set; } = NearestNeighborsDistance.L2;

        /// <summary>
        /// Number of threads and number of KD-Tree built to sppeed up the search.
        /// </summary>
        [JsonProperty("numThreads")]
        public int? NumThreads { get; set; } = 1;

        /// <summary>
        /// Seed to distribute example over trees.
        /// </summary>
        [JsonProperty("seed")]
        public int? Seed { get; set; } = 42;

        /// <summary>
        /// Column which contains a unique identifier for each observation (optional). Type must long.
        /// </summary>
        [JsonProperty("colId")]
        public string ColId { get; set; }


        public sealed class Output : Microsoft.ML.Runtime.EntryPoints.CommonOutputs.IMulticlassClassificationOutput, Microsoft.ML.Runtime.EntryPoints.CommonOutputs.ITrainerOutput
        {
            /// <summary>
            /// The trained model
            /// </summary>
            public Var<Microsoft.ML.Runtime.EntryPoints.IPredictorModel> PredictorModel { get; set; } = new Var<Microsoft.ML.Runtime.EntryPoints.IPredictorModel>();

        }
        public Var<IDataView> GetInputData() => TrainingData;

        public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
        {
            if (previousStep != null)
            {
                if (!(previousStep is ILearningPipelineDataStep dataStep))
                {
                    throw new InvalidOperationException($"{ nameof(NearestNeighborsMultiClass)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                TrainingData = dataStep.Data;
            }
            Output output = EntryPointsNearestNeighborsMultiClassHelper.Add(experiment, this);
            return new NearestNeighborsMultiClassPipelineStep(output);
        }

        private class NearestNeighborsMultiClassPipelineStep : ILearningPipelinePredictorStep
        {
            public NearestNeighborsMultiClassPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<IPredictorModel> Model { get; }
        }
    }

    #endregion
}

