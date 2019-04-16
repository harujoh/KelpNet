using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public abstract class ComputeBufferBase<T> : ComputeMemory where T : struct
    {
        public long Count { get; private set; }

        protected ComputeBufferBase(ComputeContext context, ComputeMemoryFlags flags)
            : base(context, flags)
        { }

        protected void Init()
        {
            SetID(Handle.Value);

            Size = (long)GetInfo<CLMemoryHandle, ComputeMemoryInfo, IntPtr>(Handle, ComputeMemoryInfo.Size, CL10.GetMemObjectInfo);
            Count = Size / Marshal.SizeOf(typeof(T));

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }
    }
}