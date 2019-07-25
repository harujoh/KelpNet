using System;
using Cloo.Bindings;

namespace Cloo
{
    public abstract class ComputeBufferBase<T> : ComputeMemory where T : unmanaged
    {
        public long Count { get; private set; }

        protected unsafe void Init()
        {
            SetID(Handle.Value);

            Size = (long)GetInfo<CLMemoryHandle, ComputeMemoryInfo, IntPtr>(Handle, ComputeMemoryInfo.Size, CL10.GetMemObjectInfo);
            Count = Size / sizeof(T);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }
    }
}