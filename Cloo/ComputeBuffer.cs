using System;
using System.Runtime.InteropServices;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeBuffer<T> : ComputeBufferBase<T> where T : struct
    {
        public ComputeBuffer(ComputeContext context, ComputeMemoryFlags flags, long count) : this(context, flags, count, IntPtr.Zero)
        { }

        public ComputeBuffer(ComputeContext context, ComputeMemoryFlags flags, long count, IntPtr dataPtr)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateBuffer(context.Handle, flags, new IntPtr(Marshal.SizeOf(typeof(T)) * count), dataPtr, out error);
            ComputeException.ThrowOnError(error);
            Init();
        }

        public ComputeBuffer(ComputeContext context, ComputeMemoryFlags flags, T[] data)
        {
            GCHandle dataPtr = GCHandle.Alloc(data, GCHandleType.Pinned);

            try
            {
                ComputeErrorCode error = ComputeErrorCode.Success;
                Handle = CL10.CreateBuffer(context.Handle, flags, new IntPtr(Marshal.SizeOf(typeof(T)) * data.Length), dataPtr.AddrOfPinnedObject(), out error);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                dataPtr.Free();
            }

            Init();
        }
    }
}