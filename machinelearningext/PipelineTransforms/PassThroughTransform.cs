// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;


// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using PassThroughTransform = Scikit.ML.PipelineTransforms.PassThroughTransform;

[assembly: LoadableClass(PassThroughTransform.Summary, typeof(PassThroughTransform),
    typeof(PassThroughTransform.Arguments), typeof(SignatureDataTransform),
    "Describe Transform", PassThroughTransform.LoaderSignature, "Pass", "PassThrough", "DumpView")]

[assembly: LoadableClass(PassThroughTransform.Summary, typeof(PassThroughTransform),
    null, typeof(SignatureLoadDataTransform),
    "Pass or Dump Transform", PassThroughTransform.LoaderSignature, "Pass", "PassThrough", "DumpView")]


namespace Scikit.ML.PipelineTransforms
{
    /// <summary>
    /// Inserts a transform which does nothing just to get a transform pointer.
    /// </summary>
    public class PassThroughTransform : IDataTransform
    {
        /// <summary>
        /// A unique signature.
        /// </summary>
        public const string LoaderSignature = "PassThroughTransform";  // Not more than 24 letters.
        public const string Summary = "Insert a transform which does nothing just to get a transform pointer. It can be used to dump a view on disk.";
        public const string RegistrationName = LoaderSignature;

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PASSTHRO",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Save on disk?", ShortName = "s")]
            public bool saveOnDisk = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Filename if saved.", ShortName = "f")]
            public string filename = null;

            [Argument(ArgumentType.Multiple, HelpText = "Saver settings if data is saved on disk (default is binary).", ShortName = "saver")]
            //public SubComponent<IDataSaver, SignatureDataSaver> saverSettings = new SubComponent<IDataSaver, SignatureDataSaver>("binary");
            public string saverSettings = "binary";

            public string GetSaverSettings()
            {
                var saver = GetSaverComponent();
                var res = string.Format("{0}{{{1}}}", saver.Kind, saver.SubComponentSettings);
                res = res.Replace("{}", "");
                return res;
            }

            public SubComponent<IDataSaver, SignatureDataSaver> GetSaverComponent()
            {
                return new SubComponent<IDataSaver, SignatureDataSaver>(saverSettings);
            }

            public void Save(ModelSaveContext ctx)
            {
                ctx.Writer.Write((byte)(saveOnDisk ? 1 : 0));
                ctx.Writer.Write(string.IsNullOrEmpty(filename) ? "" : filename);
                var saver = GetSaverSettings();
                ctx.Writer.Write(saver);
            }

            public void Read(ModelLoadContext ctx)
            {
                saveOnDisk = ctx.Reader.ReadByte() == 1;
                filename = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(filename))
                    filename = null;
                saverSettings = ctx.Reader.ReadString();
            }
        }

        [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
        public class ArgumentsEntryPoint : Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Input dataset",
                      Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public IDataView Data;
        }

        IDataView _input;
        Arguments _args;
        IHost _host;
        object _lock;
        bool _saved;

        public IDataView Source { get { return _input; } }

        public PassThroughTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            _input = input;
            _args = args;
            _lock = new object();
            _saved = false;
            if (_args.saveOnDisk && string.IsNullOrEmpty(_args.filename))
                throw _host.Except("If saveOnDisk is true, filename cannot be empty.");
        }

        public static PassThroughTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new PassThroughTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Save(ctx);
        }

        private PassThroughTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx);
            _lock = new object();
            _saved = false;
        }

        public ISchema Schema { get { return _input.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.AssertValue(_input, "_input");
            DumpView();
            return Source.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.AssertValue(_input, "_input");
            DumpView();
            return Source.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        public void DumpView()
        {
            if (!_args.saveOnDisk)
                return;
            lock (_lock)
            {
                if (_saved)
                    return;

                using (var ch = _host.Start("Dump View"))
                {
                    ch.Info("Dump view into '{0}'{1}.", _args.filename, File.Exists(_args.filename) ? "(overwriting)" : "");
                    var saver = ComponentCreation.CreateSaver(_host, _args.GetSaverSettings());

                    var columnsList = new List<int>();
                    for (int i = 0; i < _input.Schema.ColumnCount; ++i)
                        columnsList.Add(saver.IsColumnSavable(_input.Schema.GetColumnType(i)) && _input.Schema.IsHidden(i) ? i : -1);
                    var columns = columnsList.Where(c => c >= 0).ToArray();
                    ch.Info("Save columns: {0}", string.Join(", ", columns.Select(c => c.ToString())));
                    using (var fs2 = File.Create(_args.filename))
                        saver.SaveData(fs2, _input, columns);

                    long length = new FileInfo(_args.filename).Length;
                    ch.Info("Done dump. Size: {0}", length);
                    ch.Done();
                }

                _saved = true;
            }
        }
    }
}
