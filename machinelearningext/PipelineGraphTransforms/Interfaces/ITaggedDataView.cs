//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Scikit.ML.PipelineTransforms;


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// A view in a pipeline can be tagged.
    /// Tagged view can retrieve and put back in the pipeline.
    /// </summary>
    public interface ITaggedDataView : IDataView
    {
        /// <summary>
        /// Retrieve all views.
        /// </summary>
        /// <param name="recursive">look into previous views. Works only if it is a transform.</param>
        IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive = true);

        /// <summary>
        /// When the user selects another view, we need to keep a pointer on this view and the view set aside.
        /// </summary>
        IEnumerable<Tuple<string, ITaggedDataView>> ParallelViews { get; }

        /// <summary>
        /// References new tagged views.
        /// </summary>
        void AddRange(IEnumerable<Tuple<string, ITaggedDataView>> tagged);

        /// <summary>
        /// A tag view can host a predictor.
        /// </summary>
        IPredictor TaggedPredictor { get; }
    }

    /// <summary>
    /// Helpers for tags.
    /// </summary>
    public static class TagHelper
    {
        /// <summary>
        /// Implements get tagged view.
        /// </summary>
        /// <param name="recursive"></param>
        /// <param name="view"></param>
        public static IEnumerable<Tuple<string, ITaggedDataView>> EnumerateTaggedView(bool recursive,
                            IDataView view)
        {
            var taggedCheck = new Dictionary<string, ITaggedDataView>();
            var taggedList = new List<IDataView>();
            taggedList.Add(view);
            int position = 0;
            while (position < taggedList.Count)
            {
                view = taggedList[position];
                ++position;

                var taggedView = view as ITaggedDataView;
                if (taggedView != null && taggedView.ParallelViews != null)
                {
                    foreach (var pv in taggedView.ParallelViews)
                    {
                        if (taggedCheck.ContainsKey(pv.Item1))
                        {
                            if (!taggedCheck[pv.Item1].Equals(pv.Item2))
                                throw Contracts.Except("Tag '{0}' is used to tagged more than one view", pv.Item1);
                            continue;
                        }
                        taggedCheck[pv.Item1] = pv.Item2;
                        yield return pv;

                        if (recursive)
                            taggedList.Add(pv.Item2);
                    }
                }

                if (recursive)
                {
                    var xf = view as IDataTransform;
                    if (xf != null && xf.Source != null)
                        taggedList.Add(xf.Source);
                    var ov = view as SemiOpaqueDataView;
                    if (ov != null && ov.SourceTags != null)
                        taggedList.Add(ov.SourceTags);
                    var xa = view as AbstractSimpleTransformTemplate;
                    if (xa != null && xa.SourceEnd != null)
                        taggedList.Add(xa.SourceEnd);
                }
            }
        }

        /// <summary>
        /// Looks for a predictor among tagged predictors.
        /// If the tag ends by .zip, it assumes it is a file.
        /// </summary>
        /// <param name="env"></param>
        /// <param name="input"></param>
        /// <param name="tag"></param>
        /// <returns></returns>
        public static IPredictor GetTaggedPredictor(IHostEnvironment env, IDataView input, string tag)
        {
            if (string.IsNullOrEmpty(tag))
                throw env.Except("tag must not be null.");
            if (tag.EndsWith(".zip"))
            {
                using (Stream modelStream = new FileStream(tag, FileMode.Open, FileAccess.Read))
                {
                    var pred = ComponentCreation.LoadPredictorOrNull(env, modelStream);
#pragma warning disable CS0618
                    var obj = pred.GetPredictorObject() as IPredictor;
#pragma warning restore CS0618
                    var ipred = obj as IPredictor;
                    return ipred;
                }
            }
            else
            {
                var tagged = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == tag);
                if (!tagged.Any())
                    throw env.Except("Unable to find any view with tag '{0}'.", tag);
                if (tagged.Skip(1).Any())
                {
                    var allTagged = TagHelper.EnumerateTaggedView(true, input).ToArray();
                    throw env.Except("Ambiguous tag '{0}' - {1}/{2}.", tag,
                        allTagged.Where(c => c.Item1 == tag).Count(), allTagged.Length);
                }
                var predictor = tagged.First().Item2.TaggedPredictor;
                if (predictor == null)
                    env.Except("Tagged view '{0}' does not host a predictor.", tag);
                return predictor;
            }
        }

        /// <summary>
        /// Implements get tagged view.
        /// </summary>
        public static List<Tuple<string, ITaggedDataView>> Reconcile(IEnumerable<Tuple<string, ITaggedDataView>> viewSet)
        {
            foreach (var pair in viewSet)
            {
                var tag = pair.Item2 as ITaggedDataView;
                if (tag == null)
                    throw Contracts.Except("Tagged view '{0}' is not ITaggedDataView.", pair.Item1);
            }
            var tagged = new Dictionary<string, ITaggedDataView>();
            foreach (var item in viewSet)
            {
                if (!tagged.ContainsKey(item.Item1))
                    tagged[item.Item1] = item.Item2;
                else
                {
                    var o = tagged[item.Item1];
                    if (!o.Equals(item.Item2))
                        throw Contracts.Except("Tags cannot be reconciled for tag='{0}'.", item.Item1);
                }
            }
            return tagged.Select(c => new Tuple<string, ITaggedDataView>(c.Key, c.Value)).ToList();
        }
    }
}
