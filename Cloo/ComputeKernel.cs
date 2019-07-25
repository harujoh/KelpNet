using System;
using System.Runtime.InteropServices;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeKernel : ComputeResource
    {
        public CLKernelHandle Handle
        {
            get;
            protected set;
        }

        internal ComputeKernel(string functionName, ComputeProgram program)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateKernel(program.Handle, functionName, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void SetArgument(int index, IntPtr dataSize, IntPtr dataAddr)
        {
            ComputeErrorCode error = CL10.SetKernelArg(Handle, index, dataSize, dataAddr);
            ComputeException.ThrowOnError(error);
        }

        public void SetMemoryArgument(int index, ComputeMemory memObj)
        {
            SetValueArgument(index, memObj.Handle);
        }

        public void SetValueArgument<T>(int index, T data) where T : struct
        {
            GCHandle gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);

            try
            {
                SetArgument(
                    index,
                    new IntPtr(Marshal.SizeOf(typeof(T))),
                    gcHandle.AddrOfPinnedObject());
            }
            finally
            {
                gcHandle.Free();
            }
        }

        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseKernel(Handle);
                Handle.Invalidate();
            }
        }
    }
}