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
using ULabelToR4LabelTransform = Scikit.ML.PipelineTransforms.ULabelToR4LabelTransform;
using EntryPointULabelToR4Label = Scikit.ML.EntryPoints.EntryPointULabelToR4Label;
using EP_ULabelToR4Label = Scikit.ML.EntryPoints.ULabelToR4Label;

[assembly: LoadableClass(typeof(void), typeof(EntryPointULabelToR4Label), null,
    typeof(SignatureEntryPointModule), EP_ULabelToR4Label.Name)]


namespace Scikit.ML.EntryPoints
{
    #region Definition

    public static class EntryPointULabelToR4Label
    {
        [TlcModule.EntryPoint(Name = EntryPointsConstants.EntryPointPrefix + EP_ULabelToR4Label.Name,
                              Desc = ULabelToR4LabelTransform.Summary,
                              UserName = EP_ULabelToR4Label.Name)]
        public static CommonOutputs.TransformOutput ULabelToR4Label(IHostEnvironment env, ULabelToR4LabelTransform.ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, EP_ULabelToR4Label.Name, input);
            var view = new ULabelToR4LabelTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsULabelToR4LabelHelper
    {
        public static ULabelToR4Label.Output Add(this Microsoft.ML.Runtime.Experiment exp, ULabelToR4Label input)
        {
            var output = new ULabelToR4Label.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, ULabelToR4Label input, ULabelToR4Label.Output output)
        {
            exp.AddSerialize(EntryPointsConstants.EntryPointPrefix + EP_ULabelToR4Label.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

    /// <summary>
    /// Converts a Key label into a Float label (does nothing if the input is a float).
    /// </summary>
    public sealed partial class ULabelToR4Label : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Microsoft.ML.ILearningPipelineItem
    {
        public const string Name = nameof(ULabelToR4Label);

        public ULabelToR4Label()
        {
        }

        public ULabelToR4Label(params string[] inputColumnss)
        {
            if (inputColumnss != null)
            {
                foreach (string input in inputColumnss)
                {
                    AddColumns(input);
                }
            }
        }

        public ULabelToR4Label(params (string inputColumn, string outputColumn)[] inputOutputColumnss)
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
            var list = Columns == null ? new List<Scikit.ML.EntryPoints.Column1x1>() : new List<Scikit.ML.EntryPoints.Column1x1>(Columns);
            list.Add(OneToOneColumn<Scikit.ML.EntryPoints.Column1x1>.Create(inputColumn));
            Columns = list.ToArray();
        }

        public void AddColumns(string outputColumn, string inputColumn)
        {
            var list = Columns == null ? new List<Scikit.ML.EntryPoints.Column1x1>() : new List<Scikit.ML.EntryPoints.Column1x1>(Columns);
            list.Add(OneToOneColumn<Scikit.ML.EntryPoints.Column1x1>.Create(outputColumn, inputColumn));
            Columns = list.ToArray();
        }


        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Columns to convert.
        /// </summary>
        [JsonProperty("columns")]
        public Column1x1[] Columns { get; set; }


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
                    throw new InvalidOperationException($"{ nameof(ULabelToR4Label)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsULabelToR4LabelHelper.Add(experiment, this);
            return new ULabelToR4LabelPipelineStep(output);
        }

        private class ULabelToR4LabelPipelineStep : ILearningPipelineDataStep
        {
            public ULabelToR4LabelPipelineStep(Output output)
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

