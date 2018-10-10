// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;


namespace Scikit.ML.ScikitAPI
{
    public class ScikitPipeline : IDisposable
    {
        public class StepTransform
        {
            public string transformSettings;
            public IDataTransform transform;
        }

        public class StepPredictor
        {
            public string trainerSettings;
            public ITrainerExtended trainer;
            public IPredictor predictor;
            public RoleMappedData roleMapData;
        }

        private IHostEnvironment _env;
        private StepTransform[] _transforms;
        private StepPredictor _predictor;
        private string _loaderSettings;
        private List<KeyValuePair<RoleMappedSchema.ColumnRole, string>> _roles;
        private bool _dispose;

        public ScikitPipeline(string[] transforms = null,
                              string predictor = null,
                              IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            _transforms = new StepTransform[transforms == null ? 1 : transforms.Length + 1];
            // We add a PassThroughTransform to be able to change the source.
            _transforms[0] = new StepTransform() { transformSettings = "pass", transform = null };
            if (transforms != null)
                for (int i = 0; i < transforms.Length; ++i)
                    _transforms[i + 1] = new StepTransform() { transformSettings = transforms[i], transform = null };
            _predictor = predictor == null ? null : new StepPredictor()
            {
                trainerSettings = predictor,
                predictor = null,
                trainer = null,
                roleMapData = null
            };
            _loaderSettings = null;
            _roles = null;
        }

        public void Dispose()
        {
            if (_dispose)
            {
                (_env as ConsoleEnvironment).Dispose();
                _env = null;
            }
        }

        private IHostEnvironment ExtendedConsoleEnvironment()
        {
            _dispose = true;
            var env = new ConsoleEnvironment();
            ComponentHelper.AddStandardComponents(env);
            return env;
        }

        public ScikitPipeline(string filename, IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            using (var st = File.OpenRead(filename))
                Load(st);
        }

        public ScikitPipeline(Stream st, IHostEnvironment host = null)
        {
            _dispose = false;
            _env = host ?? ExtendedConsoleEnvironment();
            Load(st);
        }

        public void Load(Stream fs)
        {
            var transformPipe = ModelFileUtils.LoadPipeline(_env, fs, new MultiFileSource(null), true);
            var pred = _env.LoadPredictorOrNull(fs);

            IDataView root;
            var stack = new List<IDataView>();
            for (root = transformPipe; root is IDataTransform; root = ((IDataTransform)root).Source)
                stack.Add(root);
            stack.Reverse();
            if (!(stack[0] is PassThroughTransform))
                stack.Insert(0, new PassThroughTransform(_env, new PassThroughTransform.Arguments(), root));
            _transforms = new StepTransform[stack.Count];
            for (int i = 0; i < _transforms.Length; ++i)
                _transforms[i] = new StepTransform() { transform = stack[i] as IDataTransform, transformSettings = null };

            if (pred == null)
                _predictor = new StepPredictor() { predictor = null, roleMapData = null, trainer = null, trainerSettings = null };
            else
            {
#pragma warning disable CS0618
                var ipred = pred.GetPredictorObject() as IPredictor;
#pragma warning restore CS0618
                _roles = ModelFileUtils.LoadRoleMappingsOrNull(_env, fs).ToList();
                var data = new RoleMappedData(transformPipe, _roles);
                _predictor = new StepPredictor() { predictor = ipred, roleMapData = data, trainer = null, trainerSettings = null };
            }
        }

        public ScikitPipeline Train(string loaderSettings, string filename,
                                    string feature = "Feature", string label = null,
                                    string weight = null, string groupId = null)
        {
            _loaderSettings = loaderSettings;
            var loader = _env.CreateLoader(loaderSettings, new MultiFileSource(filename));
            return Train(loader);
        }

        public ScikitPipeline Train(IDataView data,
                                    string feature = "Feature", string label = null,
                                    string weight = null, string groupId = null)
        {
            IDataView trans = data;
            using (var ch = _env.Start("Create transforms"))
            {
                for (int i = 0; i < _transforms.Length; ++i)
                {
                    try
                    {
                        trans = _env.CreateTransform(_transforms[i].transformSettings, trans);
                    }
                    catch (Exception e)
                    {
                        if (e.ToString().Contains("Unknown loadable class"))
                        {
                            var nn = _env.ComponentCatalog.GetAllClasses().Length;
                            var filt = _env.ComponentCatalog.GetAllClasses()
                                                            .Select(c => c.UserName)
                                                            .OrderBy(c => c)
                                                            .Where(c => c.Trim().Length > 2);
                            var regis = string.Join("\n", filt);
                            throw Contracts.Except(e, $"Unable to create transform '{_transforms[i].transformSettings}', assembly not registered among {nn}\n{regis}");
                        }
                        throw e;
                    }
                    _transforms[i].transform = trans as IDataTransform;
                }
            }

            if (_predictor != null)
            {
                using (var ch = _env.Start("Create Predictor"))
                {
                    _predictor.trainer = TrainerHelper.CreateTrainer(_env, _predictor.trainerSettings);
                    _roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
                    _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, feature));
                    if (!string.IsNullOrEmpty(label))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, label));
                    if (!string.IsNullOrEmpty(groupId))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Group, groupId));
                    if (!string.IsNullOrEmpty(weight))
                        _roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Weight, weight));
                    var roleMap = new RoleMappedData(trans, label, feature, group: groupId, weight: weight);
                    _predictor.predictor = _predictor.trainer.Train(_env, ch, roleMap);
                    _predictor.roleMapData = roleMap;
                }
            }
            else
            {
                _predictor = new StepPredictor()
                {
                    predictor = null,
                    trainer = null,
                    trainerSettings = null,
                    roleMapData = new RoleMappedData(trans)
                };

                // We predict one to make sure everything works fine.
                using (var ch = _env.Start("Compute one prediction."))
                {
                    var df = DataFrameIO.ReadView(trans, 1, keepVectors: true, env: _env);
                    if (df.Length == 0)
                        throw _env.ExceptEmpty("Something went wrong. The pipeline does not produce any output.");
                }
            }
            return this;
        }

        /// <summary>
        /// There is no difference between predict or transform.
        /// </summary>
        public IDataView Transform(string filename)
        {
            return Predict(filename);
        }

        public IDataView Predict(string filename)
        {
            if (string.IsNullOrEmpty(_loaderSettings))
                throw _env.Except("A dataframe was used for training, another one must be used for prediction.");
            var loader = _env.CreateLoader(_loaderSettings, new MultiFileSource(filename));
            return Predict(loader);
        }

        /// <summary>
        /// There is no difference between predict or transform.
        /// </summary>
        public IDataView Transform(IDataView data)
        {
            return Predict(data);
        }

        public IDataView Predict(IDataView data)
        {
            IDataView features = null;
            if (_transforms == null)
                features = data;
            else
            {
                var first = _transforms.First().transform;
                var pass = first as PassThroughTransform;
                pass.SetSource(data);
                features = _transforms.Last().transform;
            }

            if (_predictor == null || _predictor.predictor == null)
                return features;
            var roles = new RoleMappedData(features, _roles ?? new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>());
            return PredictorHelper.Predict(_env, _predictor.predictor, roles);
        }

        public void Save(string filename)
        {
            using (var fs = File.Create(filename))
                Save(fs);
        }

        public void Save(Stream fs)
        {
            using (var ch = _env.Start("Save Predictor"))
                TrainUtils.SaveModel(_env, ch, fs, _predictor.predictor, _predictor.roleMapData);
        }
    }
}
