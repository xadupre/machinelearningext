// See the LICENSE file in the project root for more information.

using System;

namespace Scikit.ML.PipelineHelper
{
    public struct DvText : IEquatable<DvText>, IComparable<DvText>
    {
        public static DvText NA = new DvText(string.Empty);

        public ReadOnlyMemory<char> str;

        public DvText(string v) { str = new ReadOnlyMemory<char>(v == null ? null : v.ToCharArray()); }
        public DvText(ReadOnlyMemory<char> v) { str = v; }

        public override string ToString() { return str.IsEmpty ? string.Empty : str.ToString(); }
        public bool Equals(DvText other) { return ToString() == other.ToString(); }
        public int CompareTo(DvText other) { return ToString().CompareTo(other.ToString()); }
        public void Set(ReadOnlyMemory<char> value) { str = value; }
        public void Set(DvText value) { str = value.str; }
    }
}
