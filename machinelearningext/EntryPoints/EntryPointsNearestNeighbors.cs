// See the LICENSE file in the project root for more information.

#pragma warning disable
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Newtonsoft.Json;
using System;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Ext.NearestNeighbours;


namespace Microsoft.ML.Ext.EntryPoints
{
    public static class EntryPointsNearestNeighbors
    {
        #region transform

        public static NearestNeighborsTr.Output Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsTr input)
        {
            var output = new NearestNeighborsTr.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsTr input, NearestNeighborsTr.Output output)
        {
            exp.AddSerialize("ExtNearestNeighbors.NearestNeighborsTr", input, output);
        }

        #endregion

        #region binary

        public static NearestNeighborsBc.Output Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsBc input)
        {
            var output = new NearestNeighborsBc.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, NearestNeighborsBc input, NearestNeighborsBc.Output output)
        {
            exp.AddSerialize("ExtNearestNeighbors.NearestNeighborsBc", input, output);
        }

        #endregion
    }

    #region Transform

    public enum NearestNeighborsAlgorithm
    {
        kdtree = 1
    }

    public enum NearestNeighborsWeights
    {
        uniform = 1,
        distance = 2
    }

    public enum NearestNeighborsDistance
    {
        cosine = 2,
        L1 = 3,
        L2 = 4
    }

    /// <summary>
    /// Retrieve the closest neighbors among a set of points.
    /// </summary>
    public sealed partial class NearestNeighborsTr : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Microsoft.ML.ILearningPipelineItem
    {
        public NearestNeighborsTr()
        {
        }

        public NearestNeighborsTr(string featureColumn = null, string distColumn = null, string idNeighborsColumn = null)
        {
            if (featureColumn != null)
                Column = featureColumn;
            if (distColumn != null)
                DistColumn = distColumn;
            if (idNeighborsColumn != null)
                IdNeighborsColumn = idNeighborsColumn;
        }

        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Feature column
        /// </summary>
        [JsonProperty("column")]
        public string Column { get; set; } = "Features";

        /// <summary>
        /// Distance columns (output)
        /// </summary>
        [JsonProperty("distColumn")]
        public string DistColumn { get; set; } = "Distances";

        /// <summary>
        /// Id of the neighbors (output)
        /// </summary>
        [JsonProperty("idNeighborsColumn")]
        public string IdNeighborsColumn { get; set; } = "idNeighbors";

        /// <summary>
        /// Label (unused) in this transform but could be leveraged later.
        /// </summary>
        [JsonProperty("labelColumn")]
        public string LabelColumn { get; set; }

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


        public sealed class Output : Microsoft.ML.Runtime.EntryPoints.CommonOutputs.ITransformOutput
        {
            /// <summary>
            /// Transformed dataset
            /// </summary>
            public Var<Microsoft.ML.Runtime.Data.IDataView> OutputData { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

            /// <summary>
            /// Transform model
            /// </summary>
            public Var<Microsoft.ML.Runtime.EntryPoints.ITransformModel> Model { get; set; } = new Var<Microsoft.ML.Runtime.EntryPoints.ITransformModel>();

        }
        public Var<IDataView> GetInputData() => Data;

        public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
        {
            if (previousStep != null)
            {
                if (!(previousStep is ILearningPipelineDataStep dataStep))
                {
                    throw new InvalidOperationException($"{ nameof(NearestNeighborsTransform)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsNearestNeighbors.Add(experiment, this);
            return new NearestNeighborsTrPipelineStep(output);
        }

        private class NearestNeighborsTrPipelineStep : ILearningPipelineDataStep
        {
            public NearestNeighborsTrPipelineStep(Output output)
            {
                Data = output.OutputData;
                Model = output.Model;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model { get; }
        }
    }

    #endregion

    #region Binary

    /// <summary>
    /// k-Nearest Neighbors trainer for Binary Classification
    /// </summary>
    public sealed partial class NearestNeighborsBc : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInput, Microsoft.ML.ILearningPipelineItem
    {
        public NearestNeighborsBc()
        {
        }

        public NearestNeighborsBc(string featureColumn = null, string labelColumn = null)
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


        public sealed class Output : Microsoft.ML.Runtime.EntryPoints.CommonOutputs.IBinaryClassificationOutput, Microsoft.ML.Runtime.EntryPoints.CommonOutputs.ITrainerOutput
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
                    throw new InvalidOperationException($"{ nameof(NearestNeighborsBc)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                TrainingData = dataStep.Data;
            }
            Output output = EntryPointsNearestNeighbors.Add(experiment, this);
            return new NearestNeighborsBcPipelineStep(output);
        }

        private class NearestNeighborsBcPipelineStep : ILearningPipelinePredictorStep
        {
            public NearestNeighborsBcPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<IPredictorModel> Model { get; }
        }
    }

    #endregion
}

