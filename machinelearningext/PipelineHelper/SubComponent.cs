﻿// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.EntryPoints;


namespace Scikit.ML.PipelineHelper
{
    public interface ISubComponent<out TComponent> : IComponentFactory<TComponent>
    {
        TComponent CreateInstance(IHostEnvironment env, params object[] extra);
        string Kind { get; }
        string SubComponentSettings { get; }
        bool IsGood();
    }

    [Serializable]
    public class ScikitSubComponent : IEquatable<ScikitSubComponent>
    {
        private static readonly string[] _empty = new string[0];

        private string _kind;
        private string[] _settings;

        /// <summary>
        /// The type/kind of sub-component. This string will never be null, but may be empty.
        /// </summary>
        public string Kind
        {
            get { return _kind; }
            set { _kind = value ?? ""; }
        }

        /// <summary>
        /// The settings strings for the sub-component. This array will never be null, but may be empty.
        /// </summary>
        public string[] Settings
        {
            get { return _settings; }
            set { _settings = value ?? _empty; }
        }

        public string SubComponentSettings
        {
            get { return CmdParser.CombineSettings(_settings); }
            set { _settings = string.IsNullOrEmpty(value) ? _empty : new string[] { value }; }
        }

        public bool IsGood()
        {
            return true;
        }

        /// <summary>
        /// It's generally better to use the IsGood() extension method. It handles null testing
        /// and empty testing.
        /// </summary>
        public bool IsEmpty
        {
            get
            {
                AssertValid();
                return _kind.Length == 0 && _settings.Length == 0;
            }
        }

        public ScikitSubComponent()
        {
            _kind = "";
            _settings = _empty;
            AssertValid();
        }

        public ScikitSubComponent(string kind)
        {
            _kind = kind ?? "";
            _settings = _empty;
            AssertValid();
        }

        /// <summary>
        /// This assumes ownership of the settings array.
        /// </summary>
        public ScikitSubComponent(string kind, params string[] settings)
        {
            _kind = kind ?? "";
            if (settings == null || settings.Length == 1 && string.IsNullOrEmpty(settings[0]))
                settings = _empty;
            _settings = settings;
            AssertValid();
        }

        public ScikitSubComponent(string kind, string settings)
        {
            _kind = kind ?? "";
            if (string.IsNullOrEmpty(settings))
                _settings = _empty;
            else
                _settings = new string[] { settings };
            AssertValid();
        }

        private void AssertValid()
        {
            Contracts.AssertValue(_kind);
            Contracts.AssertValue(_settings);
        }

        public bool Equals(ScikitSubComponent other)
        {
            if (other == null)
                return false;
            if (_kind != other._kind)
                return false;
            if (_settings.Length != other._settings.Length)
                return false;
            for (int i = 0; i < _settings.Length; i++)
            {
                if (_settings[i] != other._settings[i])
                    return false;
            }

            return true;
        }

        public override string ToString()
        {
            if (IsEmpty)
                return "{}";

            if (_settings.Length == 0)
                return _kind;

            string str = CmdParser.CombineSettings(_settings);
            StringBuilder sb = new StringBuilder();
            CmdQuoter.QuoteValue(str, sb, true);
            return _kind + sb.ToString();
        }

        public override bool Equals(object obj)
        {
            ScikitSubComponent other = obj as ScikitSubComponent;
            if (other == null)
                return false;
            return Equals(other);
        }

        public override int GetHashCode()
        {
            int hash = Kind.GetHashCode();
            for (int i = 0; i < Settings.Length; i++)
                hash = CombineHash(hash, Settings[i].GetHashCode());
            return hash;
        }

        private static uint CombineHash(uint u1, uint u2)
        {
            return ((u1 << 7) | (u1 >> 25)) ^ u2;
        }

        private static int CombineHash(int n1, int n2)
        {
            return (int)CombineHash((uint)n1, (uint)n2);
        }

        private static void ParseCore(string str, out string kind, out string args)
        {
            kind = args = null;
            if (string.IsNullOrWhiteSpace(str))
                return;
            str = str.Trim();
            int ich = str.IndexOf('{');
            if (ich < 0)
            {
                kind = str;
                return;
            }
            if (ich == 0 || str[str.Length - 1] != '}')
                throw Contracts.Except("Invalid SubComponent string: mismatched braces, or empty component name.");

            kind = str.Substring(0, ich);
            args = CmdLexer.UnquoteValue(str.Substring(ich));
        }

        public static ScikitSubComponent Parse(string str)
        {
            string kind;
            string args;
            ParseCore(str, out kind, out args);

            Contracts.AssertValueOrNull(kind);
            Contracts.AssertValueOrNull(args);
            return new ScikitSubComponent(kind, args);
        }

        public static ScikitSubComponent<TRes, TSig> Parse<TRes, TSig>(string str)
            where TRes : class
        {
            string kind;
            string args;
            ParseCore(str, out kind, out args);

            Contracts.AssertValueOrNull(kind);
            Contracts.AssertValueOrNull(args);
            return new ScikitSubComponent<TRes, TSig>(kind, args);
        }

        public static ScikitSubComponent Create(Type type)
        {
            Contracts.Check(type != null && typeof(ScikitSubComponent).IsAssignableFrom(type));
            return (ScikitSubComponent)Activator.CreateInstance(type);
        }

        public static ScikitSubComponent Clone(ScikitSubComponent src, Type type = null)
        {
            if (src == null)
                return null;

            var dst = Create(type ?? src.GetType());
            dst._kind = src._kind;
            if (Utils.Size(src._settings) == 0)
                dst._settings = _empty;
            else
                dst._settings = (string[])src._settings.Clone();

            return dst;
        }
    }

    [Serializable]
    public class ScikitSubComponent<TRes, TSig> : ScikitSubComponent, ISubComponent<TRes>
        where TRes : class
    {
        public ScikitSubComponent()
            : base()
        {
        }

        public ScikitSubComponent(string kind)
            : base(kind)
        {
        }

        public ScikitSubComponent(string kind, params string[] settings)
            : base(kind, settings)
        {
        }

        public TRes CreateInstance(IHostEnvironment env, params object[] extra)
        {
            string options = CmdParser.CombineSettings(Settings);
            TRes result;
            if (ComponentCatalog.TryCreateInstance<TRes, TSig>(env, out result, Kind, options, extra))
                return result;
            throw Contracts.Except("Unknown loadable class: {0}", Kind).MarkSensitive(MessageSensitivity.None);
        }

        public TRes CreateComponent(IHostEnvironment env)
        {
            return CreateInstance(env);
        }
    }

    public static class SubComponentExtensions
    {
        public static bool IsGood(this ScikitSubComponent src)
        {
            return src != null && !src.IsEmpty;
        }
    }
}
