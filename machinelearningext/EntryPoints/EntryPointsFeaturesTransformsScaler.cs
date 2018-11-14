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
using ScalerTransform = Scikit.ML.FeaturesTransforms.ScalerTransform;
using EntryPointScaler = Scikit.ML.EntryPoints.EntryPointScaler;
using EP_Scaler = Scikit.ML.EntryPoints.Scaler;
using Legacy = Microsoft.ML.Legacy;

[assembly: LoadableClass(typeof(void), typeof(EntryPointScaler), null,
    typeof(SignatureEntryPointModule), EP_Scaler.Name)]


namespace Scikit.ML.EntryPoints
{
    #region Definition

    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public class ScalerTransform_ArgumentsEntryPoint : ScalerTransform.Arguments
    {
        [Argument(ArgumentType.Required, HelpText = "Input dataset",
                  Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;
    }

    public static class EntryPointScaler
    {
        [TlcModule.EntryPoint(Name = EP_Scaler.Name,
                              Desc = ScalerTransform.Summary,
                              UserName = EP_Scaler.Name)]
        public static CommonOutputs.TransformOutput Scaler(IHostEnvironment env, ScalerTransform_ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, EP_Scaler.Name, input);
            var view = new ScalerTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsScalerHelper
    {
        public static Scaler.Output Add(this Microsoft.ML.Runtime.Experiment exp, Scaler input)
        {
            var output = new Scaler.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, Scaler input, Scaler.Output output)
        {
            exp.AddEntryPoint(EP_Scaler.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

    public enum ScalerTransformScalerStrategy
    {
        meanVar = 0,
        minMax = 1
    }

    /// <summary>
    /// Rescales a column (only float).
    /// </summary>
    public sealed partial class Scaler : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Legacy.ILearningPipelineItem
    {
        public const string Name = EntryPointsConstants.EntryPointPrefix + nameof(Scaler);

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

        public Legacy.ILearningPipelineStep ApplyStep(Legacy.ILearningPipelineStep previousStep, Experiment experiment)
        {
            if (previousStep != null)
            {
                if (!(previousStep is Legacy.ILearningPipelineDataStep dataStep))
                {
                    throw new InvalidOperationException($"{ nameof(Scaler)} only supports an { nameof(Legacy.ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsScalerHelper.Add(experiment, this);
            return new ScalerPipelineStep(output);
        }

        private class ScalerPipelineStep : Legacy.ILearningPipelineDataStep
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

