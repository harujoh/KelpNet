using System.Diagnostics;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public abstract class ComputeMemory : ComputeResource
    {
        public CLMemoryHandle Handle
        {
            get;
            protected set;
        }

        public long Size { get; protected set; }

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