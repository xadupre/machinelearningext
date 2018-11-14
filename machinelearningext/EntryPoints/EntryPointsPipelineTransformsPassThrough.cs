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
using Legacy = Microsoft.ML.Legacy;
using PassThroughTransform = Scikit.ML.PipelineTransforms.PassThroughTransform;
using EntryPointPassThrough = Scikit.ML.EntryPoints.EntryPointPassThrough;
using EP_PassThrough = Scikit.ML.EntryPoints.PassThrough;

[assembly: LoadableClass(typeof(void), typeof(EntryPointPassThrough), null,
    typeof(SignatureEntryPointModule), EP_PassThrough.Name)]


namespace Scikit.ML.EntryPoints
{
    #region Definition

    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public class PassThroughTransform_ArgumentsEntryPoint : PassThroughTransform.Arguments
    {
        [Argument(ArgumentType.Required, HelpText = "Input dataset",
                  Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;
    }

    public static class EntryPointPassThrough
    {
        [TlcModule.EntryPoint(Name = EP_PassThrough.Name,
                              Desc = PassThroughTransform.Summary,
                              UserName = EP_PassThrough.Name)]
        public static CommonOutputs.TransformOutput PassThrough(IHostEnvironment env, PassThroughTransform_ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, EP_PassThrough.Name, input);
            var view = new PassThroughTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsPassThroughHelper
    {
        public static PassThrough.Output Add(this Microsoft.ML.Runtime.Experiment exp, PassThrough input)
        {
            var output = new PassThrough.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, PassThrough input, PassThrough.Output output)
        {
            exp.AddEntryPoint(EP_PassThrough.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

    /// <summary>
    /// Insert a transform which does nothing just to get a transform pointer. It can be used to dump a view on disk.
    /// </summary>
    public sealed partial class PassThrough : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Legacy.ILearningPipelineItem
    {
        public const string Name = EntryPointsConstants.EntryPointPrefix + nameof(PassThrough);

        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Save on disk?
        /// </summary>
        [JsonProperty("saveOnDisk")]
        public bool SaveOnDisk { get; set; } = false;

        /// <summary>
        /// Filename if saved.
        /// </summary>
        [JsonProperty("filename")]
        public string Filename { get; set; }

        /// <summary>
        /// Saver settings if data is saved on disk (default is binary).
        /// Example: <tt>binary</tt>, <tt>text</tt>.
        /// </summary>
        [JsonProperty("saverSettings")]
        public string SaverSettings { get; set; } = "binary";

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
                    throw new InvalidOperationException($"{ nameof(PassThrough)} only supports an { nameof(Legacy.ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsPassThroughHelper.Add(experiment, this);
            return new PassThroughPipelineStep(output);
        }

        private class PassThroughPipelineStep : Legacy.ILearningPipelineDataStep
        {
            public PassThroughPipelineStep(Output output)
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

