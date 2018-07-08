// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.EntryPoints;


namespace Microsoft.ML.Ext.PipelineHelper
{
    public interface ILearnerInputBaseArguments
    {
        IDataView ITrainingData { get; }
        Optional<string> IFeatureColumn { get; }
        NormalizeOption INormalizeFeatures { get; }
        CachingOptions ICaching { get; }
    }

    public static class EntryPointsHelper
    {
        public static TOut Train<TArg, TOut>(IHostEnvironment host, TArg input,
            Func<ITrainer> createTrainer,
            Func<string> getLabel = null,
            Func<string> getWeight = null,
            Func<string> getGroup = null,
            Func<string> getName = null,
            Func<IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>>> getCustom = null,
            ICalibratorTrainerFactory calibrator = null,
            int maxCalibrationExamples = 0)
            where TArg : ILearnerInputBaseArguments
            where TOut : CommonOutputs.TrainerOutput, new()
        {
            using (var ch = host.Start("Training"))
            {
                ISchema schema = input.ITrainingData.Schema;
                var feature = LearnerEntryPointsUtils.FindColumn(ch, schema, input.IFeatureColumn);
                var label = getLabel?.Invoke();
                var weight = getWeight?.Invoke();
                var group = getGroup?.Invoke();
                var custom = getCustom?.Invoke();

                var trainer = createTrainer();

                IDataView view = input.ITrainingData;
                TrainUtils.AddNormalizerIfNeeded(host, ch, trainer, ref view, feature, input.INormalizeFeatures);

                ch.Trace("Binding columns");
                var roleMappedData = host.CreateExamples(view, feature, label, group, weight, custom);

                RoleMappedData cachedRoleMappedData = roleMappedData;
                Cache.CachingType? cachingType = null;
                switch (input.ICaching)
                {
                    case CachingOptions.Memory:
                        {
                            cachingType = Cache.CachingType.Memory;
                            break;
                        }
                    case CachingOptions.Disk:
                        {
                            cachingType = Cache.CachingType.Disk;
                            break;
                        }
                    case CachingOptions.Auto:
                        {
                            ITrainerEx trainerEx = trainer as ITrainerEx;
                            // REVIEW: we should switch to hybrid caching in future.
                            if (!(input.ITrainingData is BinaryLoader) && (trainerEx == null || trainerEx.WantCaching))
                                // default to Memory so mml is on par with maml
                                cachingType = Cache.CachingType.Memory;
                            break;
                        }
                    case CachingOptions.None:
                        break;
                    default:
                        throw ch.ExceptParam(nameof(input.ICaching), "Unknown option for caching: '{0}'", input.ICaching);
                }

                if (cachingType.HasValue)
                {
                    var cacheView = Cache.CacheData(host, new Cache.CacheInput()
                    {
                        Data = roleMappedData.Data,
                        Caching = cachingType.Value
                    }).OutputData;
                    cachedRoleMappedData = new RoleMappedData(cacheView, roleMappedData.Schema.GetColumnRoleNames());
                }

                var predictor = TrainUtils.Train(host, ch, cachedRoleMappedData, trainer, "Train", calibrator, maxCalibrationExamples);
                var output = new TOut() { PredictorModel = new PredictorModel(host, roleMappedData, input.ITrainingData, predictor) };

                ch.Done();
                return output;
            }
        }
    }
}
