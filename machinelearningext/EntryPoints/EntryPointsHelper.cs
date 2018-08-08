// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.EntryPoints;


namespace Scikit.ML.EntryPoints
{
    public interface ILearnerInputBaseArguments
    {
        IDataView ITrainingData { get; }
        Optional<string> IFeatureColumn { get; }
        Optional<string> IWeightColumn { get; }
        Optional<string> ILabelColumn { get; }
        NormalizeOption INormalizeFeatures { get; }
        CachingOptions ICaching { get; }
    }

    public class LearnerInputBaseArguments : LearnerInputBase
    {
        public LearnerInputBaseArguments(ILearnerInputBaseArguments obj)
        {
            TrainingData = obj.ITrainingData;
            FeatureColumn = obj.IFeatureColumn;
            NormalizeFeatures = obj.INormalizeFeatures;
            Caching = obj.ICaching;
        }
    }

    public static class EntryPointsHelper
    {
        public static TOut Train<TArg, TOut>(IHost host, TArg input,
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
            var parInputs = new LearnerInputBaseArguments(input);
            return LearnerEntryPointsUtils.Train<LearnerInputBaseArguments, TOut>(host, parInputs,
                            createTrainer, getLabel: getLabel, getWeight: getWeight,
                            getGroup: getGroup, getName: getName, getCustom: getCustom,
                            calibrator: calibrator, maxCalibrationExamples: maxCalibrationExamples);
        }
    }
}
