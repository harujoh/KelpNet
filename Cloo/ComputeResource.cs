using System;

namespace Cloo
{
    public abstract class ComputeResource : ComputeObject, IDisposable
    {
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
            GC.KeepAlive(this);
        }

        protected abstract void Dispose(bool manual);

        ~ComputeResource()
        {
#if DEBUG
            Trace.WriteLine(ToString() + " leaked!", "Warning");
#endif
            Dispose(false);
        }
    }
}