﻿// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.UniversalModelFormat.Onnx;
using Microsoft.ML.Model.Onnx;


namespace Scikit.ML.OnnxHelper
{
    internal sealed class ScikitOnnxNode : OnnxNode
    {
        private readonly NodeProto _node;

        public ScikitOnnxNode(NodeProto node)
        {
            Contracts.AssertValue(node);
            _node = node;
        }

        public override void AddAttribute(string argName, double value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<double> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<float> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<bool> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, long value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<long> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, ReadOnlyMemory<char> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, string[] value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<ReadOnlyMemory<char>> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<string> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, string value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, bool value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
    }
}
