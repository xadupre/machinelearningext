﻿// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using ChainTransform = Scikit.ML.PipelineGraphTransforms.ChainTransform;

[assembly: LoadableClass(ChainTransform.Summary, typeof(ChainTransform),
    typeof(ChainTransform.Arguments), typeof(SignatureDataTransform),
    "Chain Transform", ChainTransform.LoaderSignature, "ChainTrans", "Chtr")]

[assembly: LoadableClass(ChainTransform.Summary, typeof(ChainTransform),
    null, typeof(SignatureLoadDataTransform),
    "Chain Transform", ChainTransform.LoaderSignature, "ChainTrans", "Chtr")]


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Stacks multiple transforms into a single one.
    /// </summary>
    public class ChainTransform : IDataTransform
    {
        public const string LoaderSignature = "ChainTransform";  // Not more than 24 letters.
        public const string Summary = "Chain multiple transforms into a single one.";
        public const string RegistrationName = LoaderSignature;

        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CHAITRNS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "First transform", ShortName = "xf1")]
            public SubComponent<IDataTransform, SignatureDataTransform> transformType1 = null;

            [Argument(ArgumentType.Multiple, HelpText = "Second transform", ShortName = "xf2")]
            public SubComponent<IDataTransform, SignatureDataTransform> transformType2 = null;
        }

        IDataView _input;
        IDataTransform[] _dataTransforms;
        Arguments _args;
        IHost _host;

        public IDataView Source { get { return _input; } }

        public ChainTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            _host.CheckValue(input, "input");
            _input = input;
            _args = args;
            _host.CheckUserArg(_args.transformType1.IsGood(), "transformType1", "Must specify a first transform.");
            _host.CheckUserArg(_args.transformType2.IsGood(), "transformType2", "Must specify a second transform.");
            _dataTransforms = new IDataTransform[2];
            _dataTransforms[0] = _args.transformType1.CreateInstance(env, input);
            _dataTransforms[1] = _args.transformType2.CreateInstance(env, _dataTransforms[0]);
        }

        public static ChainTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ChainTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            ctx.Writer.Write(_dataTransforms.Length);
            for (int i = 0; i < _dataTransforms.Length; ++i)
                ctx.SaveModel(_dataTransforms[i], string.Format("XF{0}", i));
        }

        private ChainTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            int nb = ctx.Reader.ReadInt32();
            _host.Check(nb > 0, "nb");
            _dataTransforms = new IDataTransform[nb];
            for (int i = 0; i < _dataTransforms.Length; ++i)
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(host, out _dataTransforms[i], string.Format("XF{0}", i),
                                i == 0 ? input : _dataTransforms[i - 1]);
        }

        public ISchema Schema { get { return _dataTransforms.Last().Schema; } }
        public bool CanShuffle { get { return _dataTransforms.Select(c => c.CanShuffle).All(c => true); } }

        public long? GetRowCount(bool lazy = true)
        {
            return _dataTransforms.Last().GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.AssertValue(_dataTransforms, "_dataTransforms");
            return _dataTransforms.Last().GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.AssertValue(_dataTransforms, "_dataTransforms");
            return _dataTransforms.Last().GetRowCursorSet(out consolidator, predicate, n, rand);
        }
    }
}