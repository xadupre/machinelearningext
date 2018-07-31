// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Basic operations with predictors.
    /// </summary>
    public static class PredictorHelper
    {
        /// <summary>
        /// Computes the predictions using a default scorer.
        /// </summary>
        public static IDataScorerTransform Predict(IHostEnvironment env, IPredictor predictor, RoleMappedData data, RoleMappedSchema trainSchema = null)
        {
            return ScoreUtils.GetScorer(predictor, data, env, trainSchema);
        }
    }
}
