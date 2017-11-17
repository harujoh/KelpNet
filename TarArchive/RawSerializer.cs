using System;
using System.Runtime.InteropServices;

namespace TarArchive
{
    internal class RawSerializer<T>
    {
        public T RawDeserialize(byte[] rawData)
        {
            return RawDeserialize(rawData, 0);
        }

        public T RawDeserialize(byte[] rawData, int position)
        {
            int rawsize = Marshal.SizeOf(typeof(T));

            if (rawsize > rawData.Length)
            {
                return default(T);
            }

            IntPtr buffer = Marshal.AllocHGlobal(rawsize);

            try
            {
                Marshal.Copy(rawData, position, buffer, rawsize);

                return (T)Marshal.PtrToStructure(buffer, typeof(T));
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public byte[] RawSerialize(T item)
        {
            int rawSize = Marshal.SizeOf(typeof(T));
            byte[] rawData = new byte[rawSize];
            IntPtr buffer = Marshal.AllocHGlobal(rawSize);

            try
            {
                Marshal.StructureToPtr(item, buffer, false);
                Marshal.Copy(buffer, rawData, 0, rawSize);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }

            return rawData;
        }
    }
}
