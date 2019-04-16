using System.Diagnostics;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public abstract class ComputeMemory : ComputeResource
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeMemoryFlags flags;

        public CLMemoryHandle Handle
        {
            get;
            protected set;
        }

        public long Size { get; protected set; }

        protected ComputeMemory(ComputeContext context, ComputeMemoryFlags flags)
        {
            this.context = context;
            this.flags = flags;
        }

        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");                
#endif
                CL10.ReleaseMemObject(Handle);
                Handle.Invalidate();
            }
        }
    }
}