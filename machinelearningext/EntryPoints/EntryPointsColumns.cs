// See the LICENSE file in the project root for more information.

#pragma warning disable
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Newtonsoft.Json;
using System;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;

namespace Scikit.ML.EntryPoints
{
    public sealed partial class Column1x1 : OneToOneColumn<Column1x1>, IOneToOneColumn
    {
        /// <summary>
        /// Name of the new column
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Name of the source column
        /// </summary>
        public string Source { get; set; }
    }
}
