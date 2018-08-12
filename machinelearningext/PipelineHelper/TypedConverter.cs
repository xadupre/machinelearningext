// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Utilities;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// The class is not protected against multithreading.
    /// </summary>
    public class TypedConverters<TLabel>
    {
        bool identity;

        ValueMapper<TLabel, DvBool> mapperBL;
        ValueMapper<TLabel, byte> mapperU1;
        ValueMapper<TLabel, ushort> mapperU2;
        ValueMapper<TLabel, uint> mapperU4;
        ValueMapper<TLabel, DvInt4> mapperI4;
        ValueMapper<TLabel, float> mapperR4;

        ValueMapper<DvBool, TLabel> mapperFromBL;
        ValueMapper<byte, TLabel> mapperFromU1;
        ValueMapper<ushort, TLabel> mapperFromU2;
        ValueMapper<uint, TLabel> mapperFromU4;
        ValueMapper<DvInt4, TLabel> mapperFromI4;
        ValueMapper<float, TLabel> mapperFromR4;

        DvBool _bl;
        byte _u1;
        ushort _u2;
        uint _u4;
        float _r4;

        DataKind _kind;
        DataKind _destKind;

        public TypedConverters()
        {
            Init(SchemaHelper.GetKind<TLabel>());
        }

        public TypedConverters(DataKind destKind)
        {
            Init(destKind);
        }

        void Init(DataKind destKind)
        {
            _kind = SchemaHelper.GetKind<TLabel>();
            _destKind = destKind;

            mapperBL = null;
            mapperU1 = null;
            mapperU2 = null;
            mapperU4 = null;
            mapperI4 = null;
            mapperR4 = null;

            mapperFromBL = null;
            mapperFromU1 = null;
            mapperFromU2 = null;
            mapperFromU4 = null;
            mapperFromI4 = null;
            mapperFromR4 = null;

            switch (destKind)
            {
                case DataKind.BL:
                    mapperBL = SchemaHelper.GetConverter<TLabel, DvBool>(out identity);
                    mapperFromBL = SchemaHelper.GetConverter<DvBool, TLabel>(out identity);
                    break;
                case DataKind.U1:
                    mapperU1 = SchemaHelper.GetConverter<TLabel, byte>(out identity);
                    mapperFromU1 = SchemaHelper.GetConverter<byte, TLabel>(out identity);
                    break;
                case DataKind.U2:
                    mapperU2 = SchemaHelper.GetConverter<TLabel, ushort>(out identity);
                    mapperFromU2 = SchemaHelper.GetConverter<ushort, TLabel>(out identity);
                    break;
                case DataKind.U4:
                    mapperU4 = SchemaHelper.GetConverter<TLabel, uint>(out identity);
                    mapperFromU4 = SchemaHelper.GetConverter<uint, TLabel>(out identity);
                    break;
                case DataKind.I4:
                    var temp = SchemaHelper.GetConverter<TLabel, float>(out identity);
                    mapperI4 = (ref TLabel src, ref DvInt4 dst) =>
                    {
                        float v = 0f;
                        temp(ref src, ref v);
                        dst = (int)v;
                    };
                    var temp2 = SchemaHelper.GetConverter<float, TLabel>(out identity);
                    mapperFromI4 = (ref DvInt4 src, ref TLabel dst) =>
                    {
                        float v = (float)src;
                        temp2(ref v, ref dst);
                    };
                    break;
                case DataKind.R4:
                    mapperR4 = SchemaHelper.GetConverter<TLabel, float>(out identity);
                    mapperFromR4 = SchemaHelper.GetConverter<float, TLabel>(out identity);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kinds {0} --> {1}", _kind, _destKind);
            }
        }

        public ValueMapper<TLabel, TDest> GetMapper<TDest>()
        {
            var kind = SchemaHelper.GetKind<TDest>();
            switch (kind)
            {
                case DataKind.BL:
                    return mapperBL as ValueMapper<TLabel, TDest>;
                case DataKind.U1:
                    return mapperU1 as ValueMapper<TLabel, TDest>;
                case DataKind.U2:
                    return mapperU2 as ValueMapper<TLabel, TDest>;
                case DataKind.U4:
                    return mapperU4 as ValueMapper<TLabel, TDest>;
                case DataKind.I4:
                    return mapperI4 as ValueMapper<TLabel, TDest>;
                case DataKind.R4:
                    return mapperR4 as ValueMapper<TLabel, TDest>;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kind {0}", kind);
            }
        }

        public ValueMapper<TDest, TLabel> GetMapperFrom<TDest>()
        {
            var kind = SchemaHelper.GetKind<TDest>();
            switch (kind)
            {
                case DataKind.BL:
                    return mapperBL as ValueMapper<TDest, TLabel>;
                case DataKind.U1:
                    return mapperU1 as ValueMapper<TDest, TLabel>;
                case DataKind.U2:
                    return mapperU2 as ValueMapper<TDest, TLabel>;
                case DataKind.U4:
                    return mapperU4 as ValueMapper<TDest, TLabel>;
                case DataKind.R4:
                    return mapperR4 as ValueMapper<TDest, TLabel>;
                default:
                    throw Contracts.ExceptNotSupp("Unsupported kind {0}", kind);
            }
        }

        public void Save(ModelSaveContext ctx, TLabel value)
        {
            switch (_kind)
            {
                case DataKind.BL:
                    mapperBL(ref value, ref _bl);
                    ctx.Writer.Write(_bl.RawValue);
                    break;
                case DataKind.U1:
                    mapperU1(ref value, ref _u1);
                    ctx.Writer.Write(_u1);
                    break;
                case DataKind.U2:
                    mapperU2(ref value, ref _u2);
                    ctx.Writer.Write(_u2);
                    break;
                case DataKind.U4:
                    mapperU4(ref value, ref _u4);
                    ctx.Writer.Write(_u4);
                    break;
                case DataKind.R4:
                    mapperR4(ref value, ref _r4);
                    ctx.Writer.Write(_r4);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Not supported kind {0}", _kind);
            }
        }

        public void Read(ModelLoadContext ctx, ref TLabel res)
        {
            switch (_kind)
            {
                case DataKind.BL:
                    var b = ctx.Reader.ReadByte();
                    if (b == DvBool.True.RawValue)
                        _bl = DvBool.True;
                    else if (b == DvBool.False.RawValue)
                        _bl = DvBool.False;
                    else
                        _bl = DvBool.NA;
                    mapperFromBL(ref _bl, ref res);
                    break;
                case DataKind.U1:
                    _u1 = ctx.Reader.ReadByte();
                    mapperFromU1(ref _u1, ref res);
                    break;
                case DataKind.U2:
                    _u2 = ctx.Reader.ReadUInt16();
                    mapperFromU2(ref _u2, ref res);
                    break;
                case DataKind.U4:
                    _u4 = ctx.Reader.ReadUInt32();
                    mapperFromU4(ref _u4, ref res);
                    break;
                case DataKind.R4:
                    _r4 = ctx.Reader.ReadFloat();
                    mapperFromR4(ref _r4, ref res);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Not supported kind {0}", _kind);
            }
        }
    }
}

