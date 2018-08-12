#if false
//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System.IO;
using System.Linq;
using Microsoft.MachineLearning.CommandLine;
using Microsoft.MachineLearning.Data;
using Microsoft.MachineLearning.Model;
using Microsoft.TMSN.TMSNlearn;

using Microsoft.MachineLearning.TlcContribPipelineTransforms;
using Microsoft.MachineLearning.TlcContribHelper;

// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.MachineLearning.LoadableClassAttribute;
using TagTrainOrScoreTransform = Microsoft.MachineLearning.TlcContribWrappingPredictors.TagTrainOrScoreTransform;


[assembly: LoadableClass(TagTrainOrScoreTransform.Summary, typeof(TagTrainOrScoreTransform),
    typeof(TagTrainOrScoreTransform.Arguments), typeof(SignatureDataTransform),
    "Train and Tag and Score a Predictor", TagTrainOrScoreTransform.LoaderSignature, "TagTrainScore")]

[assembly: LoadableClass(TagTrainOrScoreTransform.Summary, typeof(TagTrainOrScoreTransform),
    null, typeof(SignatureLoadDataTransform),
    "Train and Tag and Score a Predictor", TagTrainOrScoreTransform.LoaderSignature, "TagTrainScore")]


namespace Microsoft.MachineLearning.TlcContribWrappingPredictors
{
    /// <summary>
    /// Train or / and score a predictor hosted by a tagged view.
    /// The transform can also saves the predictor.
    /// </summary>
    public class TagTrainOrScoreTransform : AbstractSimpleTransformTemplate
    {
        public const string LoaderSignature = "TagTrainScoreTransform";
        internal const string Summary = "Trains a predictor, or gets it from a tagged view, and runs it on the data.";

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TAGSTRRE",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public new class Arguments : TrainAndScoreTransform.ArgumentsBase<SignatureTrainer>
        {
            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>")]
            public SubComponent<ICalibratorTrainer, SignatureCalibrator> calibrator = new SubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int maxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>")]
            public SubComponent<IDataScorerTransform, SignatureDataScorer> scorer;

            [Argument(ArgumentType.Required, HelpText = "To tag the predictor if it is trained.", ShortName = "tag")]
            public string tag = "taggedPredictor";

            [Argument(ArgumentType.AtMostOnce, HelpText = "To save the trained model.", ShortName = "out")]
            public string outputModel = null;

            public void Write(ModelSaveContext ctx, IHost host)
            {
#if (TLC36 || TLC37 || TLC38)
                IOHelper.Write(ctx, customColumn);
                IOHelper.Write(ctx, featureColumn);
                IOHelper.Write(ctx, groupColumn);
                IOHelper.Write(ctx, labelColumn);
                IOHelper.Write(ctx, nameColumn);
                IOHelper.Write(ctx, weightColumn);
#else
                IOHelper.Write(ctx, CustomColumn);
                IOHelper.Write(ctx, FeatureColumn);
                IOHelper.Write(ctx, GroupColumn);
                IOHelper.Write(ctx, LabelColumn);
                IOHelper.Write(ctx, NameColumn);
                IOHelper.Write(ctx, WeightColumn);
#endif

                ctx.Writer.Write(maxCalibrationExamples);
                ctx.Writer.Write(tag);
                ctx.Writer.Write(string.IsNullOrEmpty(outputModel) ? string.Empty : outputModel);

                //ctx.Writer.Write(trainer.ToString());
                //ctx.Writer.Write(calibrator.ToString());
                //ctx.Writer.Write(scorer.ToString());
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
#if (TLC36 || TLC37 || TLC38)
                customColumn = IOHelper.ReadArrayKeyValuePairStringString(ctx);
                featureColumn = IOHelper.ReadString(ctx);
                groupColumn = IOHelper.ReadString(ctx);
                labelColumn = IOHelper.ReadString(ctx);
                nameColumn = IOHelper.ReadString(ctx);
                weightColumn = IOHelper.ReadString(ctx);
#else
                CustomColumn = IOHelper.ReadArrayKeyValuePairStringString(ctx);
                FeatureColumn = IOHelper.ReadString(ctx);
                GroupColumn = IOHelper.ReadString(ctx);
                LabelColumn = IOHelper.ReadString(ctx);
                NameColumn = IOHelper.ReadString(ctx);
                WeightColumn = IOHelper.ReadString(ctx);
#endif

                maxCalibrationExamples = ctx.Reader.ReadInt32();
                tag = ctx.Reader.ReadString();
                outputModel = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(outputModel))
                    outputModel = null;

                // trainer = ctx.Reader.ReadString();
                // calibrator = ctx.Reader.ReadString();
                // scorer = ctx.Reader.ReadString();
            }
        }

        readonly Arguments _args;
        IDataScorerTransform _scorer;
        ICalibratorTrainer _cali;
        IPredictor _predictor;

#if (!TLC36)
        public TagTrainOrScoreTransform(IHostEnvironment env, Arguments args, IDataView input) :
#else
        public TagTrainOrScoreTransform(Arguments args, IHostEnvironment env, IDataView input) :
#endif
            base(env, input, LoaderSignature)
        {
            _host.CheckValue(args, "args");
            _args = args;
            _cali = null;
            _scorer = null;
            _predictor = null;
#if (!TLC36)
            _sourcePipe = Create(_host, args, input, out _sourceCtx);
#else
            _sourcePipe = Create(args, _host, input, out _sourceCtx);
#endif
        }

        public override void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write(_predictor != null ? (byte)1 : (byte)0);
            ctx.Writer.Write(_cali != null ? (byte)1 : (byte)0);
            ctx.Writer.Write(_scorer != null ? (byte)1 : (byte)0);
            if (_predictor != null)
                ctx.SaveModel(_predictor, "predictor");
            if (_cali != null)
                ctx.SaveModel(_cali, "calibrator");
            if (_scorer != null)
                ctx.SaveModel(_scorer, "scorer");
        }

#if (!TLC36)
        public static TagTrainOrScoreTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
#else
        public static TagTrainOrScoreTransform Create(ModelLoadContext ctx, IHostEnvironment env, IDataView input)
#endif
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
#if (!TLC36)
            return h.Apply("Loading Model", ch => new TagTrainOrScoreTransform(h, ctx, input));
#else
            return h.Apply("Loading Model", ch => new TagTrainOrScoreTransform(ctx, h, input));
#endif
        }

#if (!TLC36)
        public TagTrainOrScoreTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, ctx, input, LoaderSignature)
#else
        public TagTrainOrScoreTransform(ModelLoadContext ctx, IHost host, IDataView input) :
            base(ctx, host, input, LoaderSignature)
#endif
        {
            _args = new Arguments();
            _args.Read(ctx, _host);

            bool hasPredictor = ctx.Reader.ReadByte() == 1;
            bool hasCali = ctx.Reader.ReadByte() == 1;
            bool hasScorer = ctx.Reader.ReadByte() == 1;

            if (hasPredictor)
            {
#if (!TLC36)
                ctx.LoadModel<IPredictor, SignatureLoadModel>(host, out _predictor, "predictor");
#else
                ctx.LoadModel<IPredictor, SignatureLoadModel>(out _predictor, "predictor", host);
#endif
            }
            else _predictor = null;

            using (var ch = _host.Start("TagTrainOrScoreTransform loading"))
            {
                var views = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == _args.tag);
                if (views.Any())
                    throw _host.Except("Tag '{0}' is already used.", _args.tag);

                var customCols = TrainUtils.CheckAndGenerateCustomColumns(_host,
#if (TLC36 || TLC37 || TLC38)
                    _args.customColumn
#else
                    _args.CustomColumn
#endif
                    );
                string feat;
                string group;
                var data = CreateDataFromArgs(ch, new OpaqueDataView(input), _args, out feat, out group);

                if (hasCali)
                {
#if (!TLC36)
                    ctx.LoadModel<ICalibratorTrainer, SignatureLoadModel>(host, out _cali, "calibrator", _predictor);
#else
                    ctx.LoadModel<ICalibratorTrainer, SignatureLoadModel>(out _cali, "calibrator", _predictor, host);
#endif
                }
                else
                    _cali = null;

                if (_cali != null)
                    throw ch.ExceptNotImpl("Calibrator is not implemented yet.");

                if (hasScorer)
                {
#if (!TLC36)
                    ctx.LoadModel<IDataScorerTransform, SignatureLoadDataTransform>(host, out _scorer, "scorer", data.Data);
#else
                    ctx.LoadModel<IDataScorerTransform, SignatureLoadDataTransform>(out _scorer, "scorer", host, data.Data);
#endif
                }
                else
                    _scorer = null;

                ch.Info("Tagging with tag '{0}'.", _args.tag);
                var ar = new TagViewTransform.Arguments { tag = _args.tag };
#if (!TLC36)
                var res = new TagViewTransform(_host, ar, _scorer, _predictor);
#else
                var res = new TagViewTransform(ar, _host, _scorer, _predictor);
#endif
                _sourcePipe = res;
                ch.Done();
            }
        }

#region training (not delayed)

#if (!TLC36)
        IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, out IDataView sourceCtx)
#else
        IDataTransform Create(Arguments args, IHostEnvironment env, IDataView input, out IDataView sourceCtx)
#endif
        {
            sourceCtx = input;
            Contracts.CheckValue(env, "env");
            env.CheckValue(args, "args");
            env.CheckValue(input, "input");
            env.CheckValue(args.tag, "tag is empty");
#if (TLC36 || TLC37 || TLC38)
            env.CheckUserArg(args.trainer.IsGood(), "trainer",
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");
#else
            env.CheckUserArg(args.Trainer.IsGood(), "trainer",
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");
#endif

            var views = TagHelper.EnumerateTaggedView(true, input).Where(c => c.Item1 == args.tag);
            if (views.Any())
                throw env.Except("Tag '{0}' is already used.", args.tag);

            var host = env.Register("TagTrainOrScoreTransform");

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
#if (TLC36 || TLC37 || TLC38)
                ITrainer trainer = args.trainer.CreateInstance(host);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.customColumn);
#else
                ITrainer trainer = args.Trainer.CreateInstance(host);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumn);
#endif
                string feat;
                string group;
                var data = CreateDataFromArgs(ch, new OpaqueDataView(input), args, out feat, out group);
#if (TLC36 || TLC37 || TLC38)
                _predictor = TrainUtils.Train(host, ch, data, trainer, args.trainer.Kind, null,
                    args.calibrator, args.maxCalibrationExamples, null);
#else
                _predictor = TrainUtils.Train(host, ch, data, trainer, args.Trainer.Kind, null,
                    args.calibrator, args.maxCalibrationExamples, null);
#endif

                if (!string.IsNullOrEmpty(args.outputModel))
                {
                    ch.Info("Saving model into '{0}'", args.outputModel);
                    using (var fs = File.Create(args.outputModel))
                        TrainUtils.SaveModel(env, ch, fs, _predictor, data);
                    ch.Info("Done.");
                }

                if (_cali != null)
                    throw ch.ExceptNotImpl("Calibrator is not implemented yet.");

                ch.Trace("Scoring");
                _scorer = PredictorHelper.CreateDefaultScorer(_host, input, feat, group, _predictor);

                ch.Info("Tagging with tag '{0}'.", args.tag);

                var ar = new TagViewTransform.Arguments { tag = args.tag };
#if (!TLC36)
                var res = new TagViewTransform(env, ar, _scorer, _predictor);
#else
                var res = new TagViewTransform(ar, env, _scorer, _predictor);
#endif
                ch.Done();
                return res;
            }
        }

        private static RoleMappedData CreateDataFromArgs<TSigTrainer>(IExceptionContext ectx, IDataView input,
            TrainAndScoreTransform.ArgumentsBase<TSigTrainer> args, out string feat, out string group)
        {
            var schema = input.Schema;
#if (TLC36 || TLC37 || TLC38)
            feat = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "featureColumn", args.featureColumn, DefaultColumnNames.Features);
            var label = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "labelColumn", args.labelColumn, DefaultColumnNames.Label);
            group = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "groupColumn", args.groupColumn, DefaultColumnNames.GroupId);
            var weight = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "weightColumn", args.weightColumn, DefaultColumnNames.Weight);
            var name = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "nameColumn", args.nameColumn, DefaultColumnNames.Name);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ectx, args.customColumn);
#else
            feat = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "FeatureColumn", args.FeatureColumn, DefaultColumnNames.Features);
            var label = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "LabelColumn", args.LabelColumn, DefaultColumnNames.Label);
            group = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "GroupColumn", args.GroupColumn, DefaultColumnNames.GroupId);
            var weight = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "WeightColumn", args.WeightColumn, DefaultColumnNames.Weight);
            var name = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, "NameColumn", args.NameColumn, DefaultColumnNames.Name);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ectx, args.CustomColumn);
#endif
            return TrainUtils.CreateExamples(input, label, feat, group, weight, name, customCols);
        }

#endregion
    }
}

#endif
