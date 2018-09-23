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
using ResampleTransform = Scikit.ML.RandomTransforms.ResampleTransform;
using EntryPointResample = Scikit.ML.EntryPoints.EntryPointResample;
using EP_Resample = Scikit.ML.EntryPoints.Resample;

[assembly: LoadableClass(typeof(void), typeof(EntryPointResample), null,
    typeof(SignatureEntryPointModule), EP_Resample.Name)]


namespace Scikit.ML.EntryPoints
{
    #region Definition

    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public class ResampleTransform_ArgumentsEntryPoint : ResampleTransform.Arguments
    {
        [Argument(ArgumentType.Required, HelpText = "Input dataset",
                  Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;
    }

    public static class EntryPointResample
    {
        [TlcModule.EntryPoint(Name = EntryPointsConstants.EntryPointPrefix + EP_Resample.Name,
                              Desc = ResampleTransform.Summary,
                              UserName = EP_Resample.Name)]
        public static CommonOutputs.TransformOutput Resample(IHostEnvironment env, ResampleTransform_ArgumentsEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, EP_Resample.Name, input);
            var view = new ResampleTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    #endregion

    #region Experiment

    public static class EntryPointsResampleHelper
    {
        public static Resample.Output Add(this Microsoft.ML.Runtime.Experiment exp, Resample input)
        {
            var output = new Resample.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, Resample input, Resample.Output output)
        {
            exp.AddEntryPoint(EntryPointsConstants.EntryPointPrefix + EP_Resample.Name, input, output);
        }
    }

    #endregion

    #region Entry Point

    /// <summary>
    /// Randomly multiplies rows, the number of multiplication per rows is draws from a Poisson Law.
    /// </summary>
    public sealed partial class Resample : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITransformInput, Legacy.ILearningPipelineItem
    {
        public const string Name = nameof(Resample);

        /// <summary>
        /// Input dataset
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> Data { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Parameter lambda of the Poison Law.
        /// </summary>
        [JsonProperty("lambda")]
        public float Lambda { get; set; } = 1f;

        /// <summary>
        /// Seed
        /// </summary>
        [JsonProperty("seed")]
        public int? Seed { get; set; }

        /// <summary>
        /// Cache the random replication. This cache holds in memory. You can disable the cache but be aware that a second consecutive run through the view will not have  the same results.
        /// </summary>
        [JsonProperty("cache")]
        public bool Cache { get; set; } = true;

        /// <summary>
        /// Class column, to resample only for a specific class, this column contains the class information (null to resample everything).
        /// </summary>
        [JsonProperty("column")]
        public string Column { get; set; }

        /// <summary>
        /// Class to resample (null for all).
        /// </summary>
        [JsonProperty("classValue")]
        public string ClassValue { get; set; }


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
                    throw new InvalidOperationException($"{ nameof(Resample)} only supports an { nameof(Legacy.ILearningPipelineDataStep)} as an input.");
                }

                Data = dataStep.Data;
            }
            Output output = EntryPointsResampleHelper.Add(experiment, this);
            return new ResamplePipelineStep(output);
        }

        private class ResamplePipelineStep : Legacy.ILearningPipelineDataStep
        {
            public ResamplePipelineStep(Output output)
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

