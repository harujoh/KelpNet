using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace KelpNet.CL.Common
{
    public abstract class ComputeObject : IEquatable<ComputeObject>
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public IntPtr handle;

        public bool Equals(ComputeObject obj)
        {
            if (obj == null)
            {
                return false;
            }

            if (!handle.Equals(obj.handle))
            {
                return false;
            }

            return true;
        }

        protected QueriedType[] GetArrayInfo<HandleType, InfoType, QueriedType>(HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate)
        {
            IntPtr bufferSizeRet;
            getInfoDelegate(handle, paramName, IntPtr.Zero, IntPtr.Zero, out bufferSizeRet);
            var buffer = new QueriedType[bufferSizeRet.ToInt64() / Marshal.SizeOf(typeof(QueriedType))];
            GCHandle gcHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);

            try
            {
                getInfoDelegate(handle, paramName, bufferSizeRet, gcHandle.AddrOfPinnedObject(), out bufferSizeRet);
            }
            finally
            {
                gcHandle.Free();
            }

            return buffer;
        }

        protected QueriedType[] GetArrayInfo<MainHandleType, SecondHandleType, InfoType, QueriedType>(MainHandleType mainHandle, SecondHandleType secondHandle, InfoType paramName, GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType> getInfoDelegate)
        {
            IntPtr bufferSizeRet;
            getInfoDelegate(mainHandle, secondHandle, paramName, IntPtr.Zero, IntPtr.Zero, out bufferSizeRet);
            var buffer = new QueriedType[bufferSizeRet.ToInt64() / Marshal.SizeOf(typeof(QueriedType))];
            GCHandle gcHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);

            try
            {
                getInfoDelegate(mainHandle, secondHandle, paramName, bufferSizeRet, gcHandle.AddrOfPinnedObject(), out bufferSizeRet);
            }
            finally
            {
                gcHandle.Free();
            }

            return buffer;
        }

        protected QueriedType GetInfo<HandleType, InfoType, QueriedType>(HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate) where QueriedType : struct
        {
            QueriedType result = new QueriedType();
            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);

            try
            {
                getInfoDelegate(handle, paramName, (IntPtr)Marshal.SizeOf(result), gcHandle.AddrOfPinnedObject(), out _);
            }
            finally
            {
                result = (QueriedType)gcHandle.Target;
                gcHandle.Free();
            }

            return result;
        }

        protected string GetStringInfo<HandleType, InfoType>(HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate)
        {
            byte[] buffer = GetArrayInfo<HandleType, InfoType, byte>(handle, paramName, getInfoDelegate);
            char[] chars = Encoding.ASCII.GetChars(buffer, 0, buffer.Length);

            return new string(chars).TrimEnd('\0');
        }

        protected string GetStringInfo<MainHandleType, SecondHandleType, InfoType>(MainHandleType mainHandle, SecondHandleType secondHandle, InfoType paramName, GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType> getInfoDelegate)
        {
            byte[] buffer = GetArrayInfo<MainHandleType, SecondHandleType, InfoType, byte>(mainHandle, secondHandle, paramName, getInfoDelegate);
            char[] chars = Encoding.ASCII.GetChars(buffer, 0, buffer.Length);

            return new string(chars).TrimEnd('\0');
        }

        protected delegate int GetInfoDelegate<HandleType, InfoType>(HandleType objectHandle, InfoType paramName, IntPtr paramValueSize, IntPtr paramValue, out IntPtr paramValueSizeRet);

        protected delegate int GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType>(MainHandleType mainObjectHandle, SecondHandleType secondaryObjectHandle, InfoType paramName, IntPtr paramValueSize, IntPtr paramValue, out IntPtr paramValueSizeRet);
    }
}