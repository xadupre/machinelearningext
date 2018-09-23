// See the LICENSE file in the project root for more information.

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
using Legacy = Microsoft.ML.Legacy;

using _NearestNeighborsTransform = Scikit.ML.NearestNeighbors.NearestNeighborsTransform;
using EntryPointNearestNeighborsTransform = Scikit.ML.EntryPoints.EntryPointNearestNeighborsTransform;
using EP_NearestNeighbors = Scikit.ML.EntryPoints.NearestNeighbors;


[assembly: LoadableClass(typeof(void), typeof(EntryPointNearestNeighborsTransform), null,
    typeof(SignatureEntryPointModule), EP_NearestNeighbors.Name)]


namespace Scikit.ML.EntryPoints
{
    #region Definition

    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public class NearestNeighborsTransform_ArgumentsEntryPoint : _NearestNeighborsTransform.Arguments
    {
        [Argument(ArgumentType.Required, HelpText = "Input dataset",
                  Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;
    }

    public static class EntryPointNearestNeighborsTransform
    {
        [TlcModule.EntryPoint(Name = EntryPointsConstants.EntryPointPrefix + EP_NearestNeighbors.Name,
                              Desc = _NearestNeighborsTransform.Summary,
                              UserName = EP_NearestNeighbors.Name)]
        public static CommonOutputs.TransformOutput NearestNeighbors(IHostEnvironment env, NearestNeighborsTransform_ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, EP_NearestNeighbors.Name, input);
            var view = new _NearestNeighborsTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsNearestNeighborsTransformHelper
    {
        public static EP_NearestNeighbors.Output Add(this Microsoft.ML.Runtime.Experiment exp, EP_NearestNeighbors input)
        {
            var output = new EP_NearestNeighbors.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, EP_NearestNeighbors input, EP_NearestNeighbors.Output output)
        {
            exp.AddEntryPoint(EntryPointsConstants.EntryPointPrefix + EP_NearestNeighbors.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

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
    public sealed partial class NearestNeighbors : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Legacy.ILearningPipelineItem
    {
        public const string Name = nameof(NearestNeighbors);

        public NearestNeighbors()
        {
        }

        public NearestNeighbors(string featureColumn = null, string distColumn = null, string idNeighborsColumn = null)
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
        [JsonProperty("weighting")]
        public NearestNeighborsWeights Weighting { get; set; } = NearestNeighborsWeights.uniform;

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

        public Legacy.ILearningPipelineStep ApplyStep(Legacy.ILearningPipelineStep previousStep, Experiment experiment)
        {
            if (previousStep != null)
            {
                if (!(previousStep is Legacy.ILearningPipelineDataStep dataStep))
                {
                    throw new InvalidOperationException($"{ nameof(NearestNeighbors)} only supports an { nameof(Legacy.ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsNearestNeighborsTransformHelper.Add(experiment, this);
            return new NearestNeighborsTransformPipelineStep(output);
        }

        private class NearestNeighborsTransformPipelineStep : Legacy.ILearningPipelineDataStep
        {
            public NearestNeighborsTransformPipelineStep(Output output)
            {
                Data = output.OutputData;
                Model = output.Model;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model { get; }
        }
    }

    #endregion    
}

