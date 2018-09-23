﻿// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;

using PrePostProcessTrainer = Scikit.ML.PipelineTraining.PrePostProcessTrainer;

[assembly: LoadableClass(PrePostProcessTrainer.Summary, typeof(PrePostProcessTrainer), typeof(PrePostProcessTrainer.Arguments),
    new[] { typeof(SignatureTrainer) }, PrePostProcessTrainer.UserNameValue,
    PrePostProcessTrainer.LoadNameValue)]


namespace Scikit.ML.PipelineTraining
{
    using CR = RoleMappedSchema.ColumnRole;

    /// <summary>
    /// Append optional transforms to preprocess and/or postprocess a trainer.
    /// </summary>
    public class PrePostProcessTrainer : TrainerBase<IPredictor>
    {
        internal const string LoadNameValue = "PrePost";
        internal const string UserNameValue = "PPTSP";
        internal const string Summary = "Append optional transforms to preprocess and/or postprocess a trainer. " +
                                        "The pipeline will execute the following pipeline pret-pre-predictor-post.";

        /// <summary>
        /// Arguments passed
        /// </summary>
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Transform which preprocesses the data (can be null)", ShortName = "pre")]
            public ISubComponent<IDataTransform> preType = null;

            [Argument(ArgumentType.Multiple, HelpText = "Transform which preprocesses the data before training and only before training (can be null)", ShortName = "pret")]
            public ISubComponent<IDataTransform> preTrainType = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cache the data preprocessed only for training (", ShortName = "c")]
            public bool cache = false;

            [Argument(ArgumentType.Multiple, HelpText = "Transform which postprocesses the data (can be null)", ShortName = "post")]
            public ISubComponent<IDataTransform> postType = null;

            [Argument(ArgumentType.Multiple, HelpText = "Predictor", ShortName = "p")]
            public ISubComponent<ITrainer> predictorType = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output column for the predictor", ShortName = "col")]
            public string outputColumn = "output";
        }

        private readonly Arguments _args;
        private ITrainer _trainer;
        private IDataTransform _preProcess;
        private IDataTransform _postProcess;
        private IDataTransform _preTrainProcess;
        private IDataTransform _predictorAsTransform;
        private IPredictor _predictor;
        private string _outputColumn;
        private string _inputColumn;
        private bool _cache;

        public PrePostProcessTrainer(IHostEnvironment env, Arguments args) : base(env, LoadNameValue)
        {
            _args = args = args ?? new Arguments();
            Contracts.CheckUserArg(_args.predictorType.IsGood(), "predictorType", "Must specify a base learner type");
            Contracts.CheckUserArg(!string.IsNullOrEmpty(args.outputColumn), "outputColumn", "outputColumn cannot be empty");
            var temp = _args.predictorType.CreateInstance(env);
            _trainer = temp as ITrainer;
            if (_trainer == null)
                env.Except(temp == null ? "Trainer cannot be cast: {0}." : "Trainer cannot be instantiated: {0}.", _args.predictorType);
            _preProcess = null;
            _postProcess = null;
            _preTrainProcess = null;
            _predictorAsTransform = null;
            _predictor = null;
            _outputColumn = _args.outputColumn;
            _cache = args.cache;
        }

        public override TrainerInfo Info => new TrainerInfo(false, false, _cache);
        public override PredictionKind PredictionKind { get { Contracts.Assert(_trainer != null); return _trainer.PredictionKind; } }

        public override IPredictor Train(TrainContext ctx)
        {
            var data = ctx.TrainingSet;
            Contracts.CheckValue(data, "data");
            Contracts.CheckValue(_trainer, "_trainer");

            IDataView view = data.Data;

            // Preprocess only for training.
            if (_args.preTrainType != null)
            {
                using (var ch2 = Host.Start("PreProcessTraining"))
                {
                    ch2.Info("Applies a preprocess only for training: {0}", _args.preTrainType);
                    _preTrainProcess = _args.preTrainType.CreateInstance(Host, view);
                    ch2.Done();
                }
                view = _preTrainProcess;
            }

            // Preprocess.
            if (_args.preType != null)
            {
                using (var ch2 = Host.Start("PreProcess"))
                {
                    ch2.Info("Applies a preprocess: {0}", _args.preType);
                    _preProcess = _args.preType.CreateInstance(Host, view);
                    ch2.Done();
                }
            }
            else
                _preProcess = new PassThroughTransform(Host, new PassThroughTransform.Arguments { }, view);
            view = _preProcess;

            // New RoleDataMapping
            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != CR.Feature.Value)
                .Where(kvp => kvp.Key.Value != CR.Group.Value)
                .Where(kvp => kvp.Key.Value != CR.Label.Value)
                .Where(kvp => kvp.Key.Value != CR.Name.Value)
                .Where(kvp => kvp.Key.Value != CR.Weight.Value);
            if (data.Schema.Feature != null)
                roles = roles.Prepend(CR.Feature.Bind(data.Schema.Feature.Name));
            if (data.Schema.Group != null)
                roles = roles.Prepend(CR.Group.Bind(data.Schema.Group.Name));
            if (data.Schema.Label != null)
                roles = roles.Prepend(CR.Label.Bind(data.Schema.Label.Name));
            if (data.Schema.Weight != null)
                roles = roles.Prepend(CR.Weight.Bind(data.Schema.Weight.Name));
            var td = new RoleMappedData(view, roles);

            // Train.
            if (_args.predictorType != null)
            {
                using (var ch2 = Host.Start("Training"))
                {
                    var sch1 = SchemaHelper.ToString(data.Schema.Schema);
                    var sch2 = SchemaHelper.ToString(td.Schema.Schema);
                    ch2.Info("Initial schema: {0}", sch1);
                    ch2.Info("Schema before training: {0}", sch2);
                    ch2.Info("Train a predictor: {0}", _args.predictorType);
                    _predictor = _trainer.Train(td);
                    ch2.Done();
                }
            }

            // Predictor as a transform.
            {
                using (var ch2 = Host.Start("Predictor as Transform"))
                {
                    ch2.Info("Creates a transfrom from a predictor");
                    _inputColumn = td.Schema.Feature.Name;
                    _predictorAsTransform = new TransformFromValueMapper(Host, _predictor as IValueMapper,
                                                        view, td.Schema.Feature.Name, _outputColumn);
                    ch2.Done();
                }
                view = _predictorAsTransform;
            }

            // Postprocess.
            if (_args.postType != null)
            {
                using (var ch2 = Host.Start("PostProcess"))
                {
                    ch2.Info("Applies a postprocess: {0}", _args.postType);
                    _postProcess = _args.postType.CreateInstance(Host, view);
                    ch2.Done();
                }
            }
            else
                _postProcess = null;
            return CreatePredictor();
        }

        IPredictor CreatePredictor()
        {
            if (_preProcess == null)
                throw Host.Except("preProcess should not be null even if it is a PassThroughTransform.");
            return new PrePostProcessPredictor(Host, _preProcess, _predictor, _inputColumn, _outputColumn, _postProcess);
        }
    }
}
