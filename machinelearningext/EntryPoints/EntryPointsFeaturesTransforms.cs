// See the LICENSE file in the project root for more information.

#pragma warning disable
using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.CommandLine;


namespace Scikit.ML.EntryPoints
{
    public static class EntryPointsFeaturesTransforms
    {
        #region Polynomial

        public static Polynomial.Output Add(this Microsoft.ML.Runtime.Experiment exp, Polynomial input)
        {
            var output = new Polynomial.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, Polynomial input, Polynomial.Output output)
        {
            exp.AddSerialize("ExtFeaturesTransforms.Polynomial", input, output);
        }

        #endregion

        #region Scaler

        public static Scaler.Output Add(this Microsoft.ML.Runtime.Experiment exp, Scaler input)
        {
            var output = new Scaler.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, Scaler input, Scaler.Output output)
        {
            exp.AddSerialize("ExtFeaturesTransforms.Scaler", input, output);
        }

        #endregion
    }

    #region Polynomial

    /// <summary>
    /// Multiplies features, build polynomial features x1, x1^2, x1x2, x2, x2^2... The output should be cached otherwise the transform will recompute the features each time it is needed. Use CacheTransform.
    /// </summary>
    public sealed partial class Polynomial : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Microsoft.ML.ILearningPipelineItem
    {
        public Polynomial()
        {
        }

        public Polynomial(params string[] inputColumnss)
        {
            if (inputColumnss != null)
            {
                foreach (string input in inputColumnss)
                {
                    AddColumns(input);
                }
            }
        }

        public Polynomial(params (string inputColumn, string outputColumn)[] inputOutputColumnss)
        {
            if (inputOutputColumnss != null)
            {
                foreach (var inputOutput in inputOutputColumnss)
                {
                    AddColumns(inputOutput.outputColumn, inputOutput.inputColumn);
                }
            }
        }

        public void AddColumns(string inputColumn)
        {
            var list = Columns == null ? new List<Column1x1>() : new List<Column1x1>(Columns);
            list.Add(OneToOneColumn<Column1x1>.Create(inputColumn));
            Columns = list.ToArray();
        }

        public void AddColumns(string outputColumn, string inputColumn)
        {
            var list = Columns == null ? new List<Column1x1>() : new List<Column1x1>(Columns);
            list.Add(OneToOneColumn<Column1x1>.Create(outputColumn, inputColumn));
            Columns = list.ToArray();
        }

        /// <summary>
        /// Features columns (a vector)
        /// </summary>
        [JsonProperty("columns")]
        public Column1x1[] Columns { get; set; }

        /// <summary>
        /// Highest degree of the polynomial features
        /// </summary>
        [JsonProperty("degree")]
        public int Degree { get; set; } = 2;

        /// <summary>
        /// Number of threads used to estimate allowed by the transform.
        /// </summary>
        [JsonProperty("numThreads")]
        public int? NumThreads { get; set; }

        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();


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
                    throw new InvalidOperationException($"{ nameof(Polynomial)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsFeaturesTransforms.Add(experiment, this);
            return new PolynomialPipelineStep(output);
        }

        private class PolynomialPipelineStep : ILearningPipelineDataStep
        {
            public PolynomialPipelineStep(Output output)
            {
                Data = output.OutputData;
                Model = output.Model;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model { get; }
        }
    }

    #endregion

    #region Scaler

    public enum ScalerTransformScalerStrategy
    {
        meanVar = 0,
        minMax = 1
    }

    /// <summary>
    /// Rescales a column (only float).
    /// </summary>
    public sealed partial class Scaler : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Microsoft.ML.ILearningPipelineItem
    {

        public Scaler()
        {
        }

        public Scaler(params string[] inputColumnss)
        {
            if (inputColumnss != null)
            {
                foreach (string input in inputColumnss)
                {
                    AddColumns(input);
                }
            }
        }

        public Scaler(params (string inputColumn, string outputColumn)[] inputOutputColumnss)
        {
            if (inputOutputColumnss != null)
            {
                foreach (var inputOutput in inputOutputColumnss)
                {
                    AddColumns(inputOutput.outputColumn, inputOutput.inputColumn);
                }
            }
        }

        public void AddColumns(string inputColumn)
        {
            var list = Columns == null ? new List<Column1x1>() : new List<Column1x1>(Columns);
            list.Add(OneToOneColumn<Column1x1>.Create(inputColumn));
            Columns = list.ToArray();
        }

        public void AddColumns(string outputColumn, string inputColumn)
        {
            var list = Columns == null ? new List<Column1x1>() : new List<Column1x1>(Columns);
            list.Add(OneToOneColumn<Column1x1>.Create(outputColumn, inputColumn));
            Columns = list.ToArray();
        }

        /// <summary>
        /// Columns to normalize.
        /// </summary>
        [JsonProperty("columns")]
        public Column1x1[] Columns { get; set; }

        /// <summary>
        /// Scaling strategy.
        /// </summary>
        [JsonProperty("scaling")]
        public ScalerTransformScalerStrategy Scaling { get; set; } = ScalerTransformScalerStrategy.meanVar;

        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();


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
                    throw new InvalidOperationException($"{ nameof(Scaler)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsFeaturesTransforms.Add(experiment, this);
            return new ScalerPipelineStep(output);
        }

        private class ScalerPipelineStep : ILearningPipelineDataStep
        {
            public ScalerPipelineStep(Output output)
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

