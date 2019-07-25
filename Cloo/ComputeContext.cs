using System;
using System.Collections.Generic;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeContext : ComputeResource
    {
        public CLContextHandle Handle
        {
            get;
            protected set;
        }

        public ComputeContext(ICollection<ComputeDevice> devices, ComputeContextPropertyList properties, ComputeContextNotifier notify, IntPtr notifyDataPtr)
        {
            int handleCount;
            CLDeviceHandle[] deviceHandles = ComputeTools.ExtractHandles(devices, out handleCount);
            IntPtr[] propertyArray = properties?.ToIntPtrArray();

            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateContext(propertyArray, handleCount, deviceHandles, notify, notifyDataPtr, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        protected override void Dispose(bool manual)
        {
            // free native resources
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseContext(Handle);
                Handle.Invalidate();
            }
        }
    }
}