using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace KelpNet.Tools
{
    public class NpyFormat
    {
        public static T Load<T>(byte[] bytes) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return ToGenericType<T>(LoadMatrix(bytes));
        }

        public static T Load<T>(byte[] bytes, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load<T>(bytes);
        }

        public static T Load<T>(string path, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load<T>(path);
        }

        public static T Load<T>(Stream stream, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load<T>(stream);
        }

        public static T Load<T>(string path) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var stream = new FileStream(path, FileMode.Open))
            {
                return Load<T>(stream);
            }
        }

        public static T Load<T>(Stream stream) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return ToGenericType<T>(LoadMatrix(stream));
        }

        public static Array LoadMatrix(byte[] bytes)
        {
            using (var stream = new MemoryStream(bytes))
            {
                return LoadMatrix(stream);
            }
        }

        public static Array LoadMatrix(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open))
            {
                return LoadMatrix(stream);
            }
        }

        public static Array LoadMatrix(Stream stream)
        {
            using (var reader = new BinaryReader(stream, Encoding.ASCII))
            {
                int bytes;
                Type type;
                int[] shape;

                if (!ParseReader(reader, out bytes, out type, out shape))
                {
                    throw new FormatException();
                }

                Array matrix = Array.CreateInstance(type, shape);

                if (type == typeof(String))
                {
                    new NotImplementedException();
                }

                return ReadValueMatrix(reader, matrix, bytes, type, shape);
            }
        }

        private static Array ReadValueMatrix(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
        {
            int total = 1;

            for (int i = 0; i < shape.Length; i++)
            {
                total *= shape[i];
            }

            var buffer = new byte[bytes * total];
            reader.Read(buffer, 0, buffer.Length);
            Buffer.BlockCopy(buffer, 0, matrix, 0, buffer.Length);

            return matrix;
        }

        private static bool ParseReader(BinaryReader reader, out int bytes, out Type t, out int[] shape)
        {
            bytes = 0;
            t = null;
            shape = null;

            if (reader.ReadChar() != 63) return false;
            if (reader.ReadChar() != 'N') return false;
            if (reader.ReadChar() != 'U') return false;
            if (reader.ReadChar() != 'M') return false;
            if (reader.ReadChar() != 'P') return false;
            if (reader.ReadChar() != 'Y') return false;

            byte major = reader.ReadByte();
            byte minor = reader.ReadByte();

            if (major != 1 || minor != 0)
            {
                throw new NotSupportedException();
            }

            ushort len = reader.ReadUInt16();
            string header = new String(reader.ReadChars(len));
            string mark = "'descr': '";
            int s = header.IndexOf(mark) + mark.Length;
            int e = header.IndexOf("'", s + 1);
            string type = header.Substring(s, e - s);
            bool? isLittleEndian;

            t = GetType(type, out bytes, out isLittleEndian);

            if (isLittleEndian.HasValue && isLittleEndian.Value == false)
            {
                throw new Exception();
            }

            mark = "'fortran_order': ";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(",", s + 1);

            bool fortran = bool.Parse(header.Substring(s, e - s));

            if (fortran)
            {
                throw new Exception();
            }

            mark = "'shape': (";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(")", s + 1);

            if (e != -1)
            {
                shape = header.Substring(s, e - s).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(Int32.Parse).ToArray();
            }
            else
            {
                shape = new[] { 0 };
            }

            return true;
        }

        private static Type GetType(string dtype, out int bytes, out bool? isLittleEndian)
        {
            isLittleEndian = IsLittleEndian(dtype);
            bytes = Int32.Parse(dtype.Substring(2));

            string typeCode = dtype.Substring(1);

            if (typeCode == "b1")
            {
                return typeof(bool);
            }

            if (typeCode == "i1")
            {
                return typeof(SByte);
            }

            if (typeCode == "i2")
            {
                return typeof(Int16);
            }

            if (typeCode == "i4")
            {
                return typeof(Int32);
            }

            if (typeCode == "i8")
            {
                return typeof(Int64);
            }

            if (typeCode == "u1")
            {
                return typeof(Byte);
            }

            if (typeCode == "u2")
            {
                return typeof(UInt16);
            }

            if (typeCode == "u4")
            {
                return typeof(UInt32);
            }

            if (typeCode == "u8")
            {
                return typeof(UInt64);
            }

            if (typeCode == "f4")
            {
                return typeof(Single);
            }

            if (typeCode == "f8")
            {
                return typeof(Double);
            }

            if (typeCode.StartsWith("S"))
            {
                return typeof(String);
            }

            throw new NotSupportedException();
        }

        private static bool? IsLittleEndian(string type)
        {
            bool? littleEndian = null;

            switch (type[0])
            {
                case '<':
                    littleEndian = true;
                    break;
                case '>':
                    littleEndian = false;
                    break;
                case '|':
                    littleEndian = null;
                    break;
                default:
                    throw new Exception();
            }

            return littleEndian;
        }


        public static T ToGenericType<T>(object value)
        {
            Type type = typeof(T);

            if (value == null)
            {
                return (T)Convert.ChangeType(null, type);
            }

            if (type.IsInstanceOfType(value))
            {
                return (T)value;
            }

            if (type.IsEnum)
            {
                return (T)Enum.ToObject(type, (int)Convert.ChangeType(value, typeof(int)));
            }

            Type inputType = value.GetType();

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
            {
                MethodInfo setter = type.GetMethod("op_Implicit", new[] { inputType });
                return (T)setter.Invoke(null, new object[] { value });
            }

            var methods = new List<MethodInfo>();
            methods.AddRange(inputType.GetMethods(BindingFlags.Public | BindingFlags.Static));
            methods.AddRange(type.GetMethods(BindingFlags.Public | BindingFlags.Static));

            foreach (MethodInfo m in methods)
            {
                if (m.IsPublic && m.IsStatic)
                {
                    if ((m.Name == "op_Implicit" || m.Name == "op_Explicit") && m.ReturnType == type)
                    {
                        ParameterInfo[] p = m.GetParameters();

                        if (p.Length == 1 && p[0].ParameterType.IsInstanceOfType(value))
                        {
                            return (T)m.Invoke(null, new[] { value });
                        }
                    }
                }
            }

            return (T)value;
        }
    }
}