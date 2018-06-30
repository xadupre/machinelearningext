// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.TestHelper
{
    public static class TestTrainerHelper
    {
        public class WrappedPredictorWithNoDistInterface : IPredictor, IValueMapper, ICanSaveModel
        {
            #region identification

            public const string LoaderSignature = "WrappedPWithNoDistI";
            public const string RegistrationName = "WrappedPWithNoDistI";

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "WRAPNODI",
                    verWrittenCur: 0x00010001,
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }

            #endregion

            IPredictor _predictor;

            public WrappedPredictorWithNoDistInterface(IPredictor pred)
            {
                _predictor = pred;
            }
            public PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }
            public IPredictor Predictor { get { return _predictor; } }
            public ColumnType InputType { get { return (_predictor as IValueMapper).InputType; } }
            public ColumnType OutputType { get { return (_predictor as IValueMapper).OutputType; } }
            public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>() { return (_predictor as IValueMapper).GetMapper<TSrc, TDst>(); }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.CheckValue(ctx, "ctx");
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());
                Contracts.CheckValue(_predictor, "_predictor");
                ctx.SaveModel(_predictor, "predictor");
            }

            private WrappedPredictorWithNoDistInterface(IHostEnvironment env, ModelLoadContext ctx)
            {
                ctx.LoadModel<IPredictor, SignatureLoadModel>(env, out _predictor, "predictor");
                Contracts.CheckValue(_predictor, "_predictor");
            }

            public static WrappedPredictorWithNoDistInterface Create(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, "env");
                env.CheckValue(ctx, "ctx");
                ctx.CheckAtModel(GetVersionInfo());
                return new WrappedPredictorWithNoDistInterface(env, ctx);
            }
        }

        /// <summary>
        /// Convert a Predictor into a IPredictor with method GetPredictorObject.
        /// As it generates a warning, all functions needing this conversion should call this
        /// function to minimize the number of raised warnings.
        /// </summary>
        public static IPredictor IPredictorFromPredictor(Predictor pred)
        {
#pragma warning disable CS0618
            var res = pred.GetPredictorObject() as IPredictor;
#pragma warning restore CS0618
            Contracts.Assert(res != null);
            return res;
        }

        /// <summary>
        /// Changes the default scorer for ExePythonPredictor.
        /// </summary>
        public static IDataScorerTransform CreateDefaultScorer(IHostEnvironment env,
                                                IDataView view, string featureColumn, string groupColumn,
                                                IPredictor ipredictor)
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            if (string.IsNullOrEmpty(featureColumn))
                throw env.Except("featureColumn cannot be null");
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, featureColumn));
            if (!string.IsNullOrEmpty(groupColumn))
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Group, groupColumn));
            var data = RoleMappedData.Create(view, roles);
            return CreateDefaultScorer(env, data, ipredictor);
        }

        /// <summary>
        /// Implements a different behaviour when the predictor inherits from 
        /// IPredictorScorer (the predictor knows the scorer to use).
        /// </summary>
        public static IDataScorerTransform CreateDefaultScorer(IHostEnvironment env,
                                                RoleMappedData roles, IPredictor ipredictor)
        {
            IDataScorerTransform scorer;
            env.CheckValue(ipredictor, "IPredictor");
            var iter = roles.Schema.GetColumnRoleNames().Where(c =>
                                        c.Key.Value != RoleMappedSchema.ColumnRole.Feature.Value &&
                                        c.Key.Value != RoleMappedSchema.ColumnRole.Group.Value);
            if (ipredictor.PredictionKind == PredictionKind.MultiClassClassification && ipredictor is IValueMapperDist)
            {
                // There is an issue with the code creating the default scorer. It expects to find a Float
                // as the output of DistType (from by IValueMapperDist)
                var newPred = new WrappedPredictorWithNoDistInterface(ipredictor);
                scorer = ScoreUtils.GetScorer(null, newPred, roles.Data, roles.Schema.Feature.Name,
                                                roles.Schema.Group == null ? null : roles.Schema.Group.Name,
                                                iter, env, null);
            }
            else
                scorer = ScoreUtils.GetScorer(null, ipredictor, roles.Data, roles.Schema.Feature.Name,
                                                    roles.Schema.Group == null ? null : roles.Schema.Group.Name,
                                                    iter, env, null);
            return scorer;
        }

        /// <summary>
        /// Finalize the test on a predictor, calls the predictor with a scorer,
        /// saves the data, saves the models, loads it back, saves the data again,
        /// checks the output is the same.
        /// </summary>
        /// <param name="env"></param>
        /// <param name="outModelFilePath"></param>
        /// <param name="predictor"></param>
        /// <param name="roles"></param>
        /// <param name="outData"></param>
        /// <param name="outData2"></param>
        /// <param name="kind">prediction kind</param>
        /// <param name="checkError">checks errors</param>
        /// <param name="ratio">check the error is below that threshold (if checkError is true)</param>
        public static void FinalizeSerializationTest(TlcEnvironment env,
                            string outModelFilePath, IPredictor predictor,
                            RoleMappedData roles, string outData, string outData2,
                            PredictionKind kind, bool checkError = true,
                            float ratio = 0.8f)
        {
            string labelColumn = kind != PredictionKind.Clustering ? roles.Schema.Label.Name : null;
            string featuresColumn = roles.Schema.Feature.Name;

            #region save, reading, running

            // Saves model.
            using (var ch = env.Start("Save"))
            {
                using (var fs = File.Create(outModelFilePath))
                    TrainUtils.SaveModel(env, ch, fs, predictor, roles);
                ch.Done();
            }
            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outModelFilePath);

            // Loads the model back.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var pred_local = env.LoadPredictorOrNull(fs);
                if (pred_local == null)
                    throw new Exception(string.Format("Unable to load '{0}'", outModelFilePath));
                if (predictor.GetType() != IPredictorFromPredictor(pred_local).GetType())
                    throw new Exception(string.Format("Type mismatch {0} != {1}", predictor.GetType(), pred_local.GetType()));
            }

            // Checks the outputs.
            var sch1 = SchemaHelper.ToString(roles.Schema.Schema);
            var scorer = CreateDefaultScorer(env, roles, predictor);

            var sch2 = SchemaHelper.ToString(scorer.Schema);
            if (string.IsNullOrEmpty(sch1) || string.IsNullOrEmpty(sch2))
                throw new Exception("Empty schemas");

            var saver = env.CreateSaver("Text");
            var columns = new int[scorer.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = saver.IsColumnSavable(scorer.Schema.GetColumnType(i)) ? i : -1;
            columns = columns.Where(c => c >= 0).ToArray();
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, scorer, columns);

            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outData);

            // Check we have the same output.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var model = env.LoadPredictorOrNull(fs);
                scorer = CreateDefaultScorer(env, roles, predictor);
                saver = env.CreateSaver("Text");
                using (var fs2 = File.Create(outData2))
                    saver.SaveData(fs2, scorer, columns);
            }

            var t1 = File.ReadAllLines(outData);
            var t2 = File.ReadAllLines(outData2);
            if (t1.Length != t2.Length)
                throw new Exception(string.Format("Not the same number of lines: {0} != {1}", t1.Length, t2.Length));
            var linesN = new List<int>();
            for (int i = 0; i < t1.Length; ++i)
            {
                if (t1[i] != t2[i])
                    linesN.Add(i);
            }
            if (linesN.Count > t1.Length * 4 / 100)
            {
                var rows = linesN.Select(i => string.Format("Mismatch on line (1) {0}/{3}:\n{1}\n{2}", i, t1[i], t2[i], t1.Length));
                throw new Exception(string.Join("\n", rows));
            }

            #endregion

            #region clustering 

            if (kind == PredictionKind.Clustering)
                // Nothing to do here.
                return;

            #endregion

            string expectedOuput = kind == PredictionKind.Regression ? "Score" : "PredictedLabel";

            // Get label and basic checking about performance.
            using (var cursor = scorer.GetRowCursor(i => true))
            {
                int ilabel, ipred;
                if (!cursor.Schema.TryGetColumnIndex(labelColumn, out ilabel))
                    throw new Exception(string.Format("Unable to find column '{0}' in {1}'", "Label", SchemaHelper.ToString(cursor.Schema)));

                if (!cursor.Schema.TryGetColumnIndex(expectedOuput, out ipred))
                    throw new Exception(string.Format("Unable to find column '{0}' in {1}'", expectedOuput, SchemaHelper.ToString(cursor.Schema)));

                var ty1 = cursor.Schema.GetColumnType(ilabel);
                var ty2 = cursor.Schema.GetColumnType(ipred);
                var dist1 = new Dictionary<int, int>();
                var dist2 = new Dictionary<int, int>();
                var conf = new Dictionary<Tuple<int, int>, long>();

                if (kind == PredictionKind.MultiClassClassification)
                {
                    #region multiclass

                    if (!ty2.IsKey)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    if (ty1.RawKind == DataKind.R4)
                    {
                        var lgetter = cursor.GetGetter<float>(ilabel);
                        var pgetter = cursor.GetGetter<uint>(ipred);
                        float ans = 0;
                        uint pre = 0;
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            // The scorer +1 to the argmax.
                            ++ans;

                            var key = new Tuple<int, int>((int)pre, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else if (ty1.RawKind == DataKind.U4 && ty1.IsKey)
                    {
                        var lgetter = cursor.GetGetter<uint>(ilabel);
                        var pgetter = cursor.GetGetter<uint>(ipred);
                        uint ans = 0;
                        uint pre = 0;
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            var key = new Tuple<int, int>((int)pre, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else
                        throw new NotImplementedException(string.Format("Not implemented for type {0}", ty1.ToString()));
                    #endregion
                }
                else if (kind == PredictionKind.BinaryClassification)
                {
                    #region binary classification

                    if (ty2.RawKind != DataKind.Bool)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    if (ty1.RawKind == DataKind.R4)
                    {
                        var lgetter = cursor.GetGetter<float>(ilabel);
                        var pgetter = cursor.GetGetter<DvBool>(ipred);
                        float ans = 0;
                        DvBool pre = default(DvBool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            if (ans != 0 && ans != 1)
                                throw Contracts.Except("The problem is not binary, expected answer is {0}", ans);

                            var key = new Tuple<int, int>(pre.IsTrue ? 1 : 0, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else if (ty1.RawKind == DataKind.U4)
                    {
                        var lgetter = cursor.GetGetter<uint>(ilabel);
                        var pgetter = cursor.GetGetter<DvBool>(ipred);
                        uint ans = 0;
                        DvBool pre = default(DvBool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);
                            if (ty1.IsKey)
                                --ans;

                            if (ans != 0 && ans != 1)
                                throw Contracts.Except("The problem is not binary, expected answer is {0}", ans);

                            var key = new Tuple<int, int>(pre.IsTrue ? 1 : 0, (int)ans);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];
                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else if (ty1.RawKind == DataKind.BL)
                    {
                        var lgetter = cursor.GetGetter<DvBool>(ilabel);
                        var pgetter = cursor.GetGetter<DvBool>(ipred);
                        DvBool ans = default(DvBool);
                        DvBool pre = default(DvBool);
                        while (cursor.MoveNext())
                        {
                            lgetter(ref ans);
                            pgetter(ref pre);

                            var key = new Tuple<int, int>(pre.IsTrue ? 1 : 0, ans.IsTrue ? 1 : 0);
                            if (!conf.ContainsKey(key))
                                conf[key] = 1;
                            else
                                ++conf[key];

                            if (!dist1.ContainsKey((int)ans))
                                dist1[(int)ans] = 1;
                            else
                                ++dist1[(int)ans];
                            if (!dist2.ContainsKey((int)pre))
                                dist2[(int)pre] = 1;
                            else
                                ++dist2[(int)pre];
                        }
                    }
                    else
                        throw new NotImplementedException(string.Format("Not implemented for type {0}", ty1));

                    #endregion
                }
                else if (kind == PredictionKind.Regression)
                {
                    #region regression

                    if (ty1.RawKind != DataKind.R4)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));
                    if (ty2.RawKind != DataKind.R4)
                        throw new Exception(string.Format("Label='{0}' Predicted={1}'\nSchema: {2}", ty1, ty2, SchemaHelper.ToString(cursor.Schema)));

                    var lgetter = cursor.GetGetter<float>(ilabel);
                    var pgetter = cursor.GetGetter<float>(ipred);
                    float ans = 0;
                    float pre = 0f;
                    float error = 0f;
                    while (cursor.MoveNext())
                    {
                        lgetter(ref ans);
                        pgetter(ref pre);
                        error += (ans - pre) * (ans - pre);
                        if (!dist1.ContainsKey((int)ans))
                            dist1[(int)ans] = 1;
                        else
                            ++dist1[(int)ans];
                        if (!dist2.ContainsKey((int)pre))
                            dist2[(int)pre] = 1;
                        else
                            ++dist2[(int)pre];
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                        throw new Exception("Regression wen wrong. Error is infinite.");

                    #endregion
                }
                else
                    throw new NotImplementedException(string.Format("Not implemented for kind {0}", kind));

                var nbError = conf.Where(c => c.Key.Item1 != c.Key.Item2).Select(c => c.Value).Sum();
                var nbTotal = conf.Select(c => c.Value).Sum();

                if (checkError && (nbError * 1.0 > nbTotal * ratio || dist2.Count <= 1))
                {
                    var sconf = string.Join("\n", conf.OrderBy(c => c.Key)
                                      .Select(c => string.Format("pred={0} exp={1} count={2}", c.Key.Item1, c.Key.Item2, c.Value)));
                    var sdist2 = string.Join("\n", dist1.OrderBy(c => c.Key)
                                       .Select(c => string.Format("label={0} count={1}", c.Key, c.Value)));
                    var sdist1 = string.Join("\n", dist2.OrderBy(c => c.Key).Take(20)
                                       .Select(c => string.Format("label={0} count={1}", c.Key, c.Value)));
                    throw new Exception(string.Format("Too many errors {0}/{1}={7}\n###########\nConfusion:\n{2}\n########\nDIST1\n{3}\n###########\nDIST2\n{4}\nOutput:\n{5}\n...\n{6}",
                                    nbError, nbTotal,
                                    sconf, sdist1, sdist2,
                                    string.Join("\n", t1.Take(Math.Min(30, t1.Length))),
                                    string.Join("\n", t1.Skip(Math.Max(0, t1.Length - 30))),
                                    nbError * 1.0 / nbTotal));
                }
            }
        }        
    }
}
