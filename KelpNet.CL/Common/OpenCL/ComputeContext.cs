using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace KelpNet.CL.Common
{
    public class ComputeContext : ComputeObject, IDisposable
    {
        public ComputeContext(ICollection<ComputeDevice> devices, ComputeContextPropertyList properties, ComputeContextNotifier notify, IntPtr notifyDataPtr)
        {
            IntPtr[] deviceHandles = ComputeTools.ExtractHandles(devices, out var handleCount);
            IntPtr[] propertyArray = properties?.ToIntPtrArray();

            handle = CL10.CreateContext(propertyArray, handleCount, deviceHandles, notify, notifyDataPtr, out _);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void Dispose()
        {
#if DEBUG
            Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
            CL10.ReleaseContext(handle);
        }
    }
}