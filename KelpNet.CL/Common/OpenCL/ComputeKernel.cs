using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace KelpNet.CL.Common
{
    public class ComputeKernel : ComputeObject, IDisposable
    {
        internal ComputeKernel(string functionName, ComputeProgram program)
        {
            handle = CL10.CreateKernel(program.handle, functionName, out _);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void SetArgument(int index, IntPtr dataSize, IntPtr dataAddr)
        {
            CL10.SetKernelArg(handle, index, dataSize, dataAddr);
        }

        public void SetMemoryArgument(int index, ComputeObject memObj)
        {
            SetValueArgument(index, memObj.handle);
        }

        public void SetValueArgument<T>(int index, T data) where T : unmanaged
        {
            GCHandle gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);

            try
            {
                SetArgument(index, new IntPtr(Unsafe.SizeOf<T>()), gcHandle.AddrOfPinnedObject());
            }
            finally
            {
                gcHandle.Free();
            }
        }

        public void Dispose()
        {
#if DEBUG
            Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
            CL10.ReleaseKernel(handle);
        }
    }
}