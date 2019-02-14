using System;
using System.IO;
using System.Linq;
using System.Text;

namespace KelpNet.Tools
{
    class BinaryReaderBE : BinaryReader
    {
        public BinaryReaderBE(Stream input)
            : base(input)
        {
        }
        public BinaryReaderBE(Stream input, Encoding encoding)
            : base(input, encoding)
        {
        }

        public override short ReadInt16()
        {
            return this._ToBigEndian(base.ReadInt16());
        }
        public override int ReadInt32()
        {
            return this._ToBigEndian(base.ReadInt32());
        }
        public override long ReadInt64()
        {
            return this._ToBigEndian(base.ReadInt64());
        }
        public override ushort ReadUInt16()
        {
            return this._ToBigEndian(base.ReadUInt16());
        }
        public override uint ReadUInt32()
        {
            return this._ToBigEndian(base.ReadUInt32());
        }
        public override ulong ReadUInt64()
        {
            return this._ToBigEndian(base.ReadUInt64());
        }
        public override float ReadSingle()
        {
            return this._ToBigEndian(base.ReadSingle());
        }
        public override double ReadDouble()
        {
            return this._ToBigEndian(base.ReadDouble());
        }
        public override decimal ReadDecimal()
        {
            throw new NotImplementedException();
        }

        private short _ToBigEndian(short value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToInt16(bytes, 0);
        }

        private ushort _ToBigEndian(ushort value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToUInt16(bytes, 0);
        }

        private int _ToBigEndian(int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        private uint _ToBigEndian(uint value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToUInt32(bytes, 0);
        }

        private long _ToBigEndian(long value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToInt64(bytes, 0);
        }

        private ulong _ToBigEndian(ulong value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToUInt64(bytes, 0);
        }

        private float _ToBigEndian(float value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToSingle(bytes, 0);
        }

        private double _ToBigEndian(double value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = this._ReverseBytes(bytes);
            return BitConverter.ToDouble(bytes, 0);
        }

        private byte[] _ReverseBytes(byte[] bytes)
        {
            if (bytes == null)
            {
                return null;
            }
            return bytes.Reverse().ToArray();
        }
    }
}
