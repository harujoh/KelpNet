using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace KelpNet.CL.Common
{
    public class ComputeProgram : ComputeObject, IDisposable
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeProgramBuildNotifier buildNotify;

        public ComputeProgram(ComputeContext context, string source)
        {
            handle = CL10.CreateProgramWithSource(context.handle, 1, new[] { source }, null, out _);

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void Build(ICollection<ComputeDevice> devices, string options, ComputeProgramBuildNotifier notify, IntPtr notifyDataPtr)
        {
            IntPtr[] deviceHandles = ComputeTools.ExtractHandles(devices, out var handleCount);
            buildNotify = notify;

            CL10.BuildProgram(handle, handleCount, deviceHandles, options, buildNotify, notifyDataPtr);
        }

        public ComputeKernel CreateKernel(string functionName)
        {
            return new ComputeKernel(functionName, this);
        }

        public string GetBuildLog(ComputeDevice device)
        {
            return GetStringInfo(handle, device.handle, ComputeProgramBuildInfo.BuildLog, CL10.GetProgramBuildInfo);
        }

        public void Dispose()
        {
#if DEBUG
            Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
            CL10.ReleaseProgram(handle);
        }
    }
}