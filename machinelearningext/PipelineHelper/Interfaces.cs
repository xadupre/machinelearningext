// See the LICENSE file in the project root for more information.

namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// The transform is trainable: it must be trained on the training data.
    /// </summary>
    public interface ITrainableTransform
    {
        void Estimate();
    }
}
