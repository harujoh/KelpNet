using System;
using System.Runtime.InteropServices;

namespace KelpNet.Tools
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
    }
}
