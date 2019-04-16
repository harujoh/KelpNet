using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeProgram : ComputeResource
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<string> source;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private string buildOptions;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeProgramBuildNotifier buildNotify;


        public CLProgramHandle Handle
        {
            get;
            protected set;
        }

        public ComputeContext Context { get { return context; } }

        public ComputeProgram(ComputeContext context, string source)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateProgramWithSource(context.Handle, 1, new string[] { source }, null, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            this.context = context;
            this.devices = context.Devices;
            this.source = new ReadOnlyCollection<string>(new string[] { source });

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void Build(ICollection<ComputeDevice> devices, string options, ComputeProgramBuildNotifier notify, IntPtr notifyDataPtr)
        {
            int handleCount;
            CLDeviceHandle[] deviceHandles = ComputeTools.ExtractHandles(devices, out handleCount);
            buildOptions = (options != null) ? options : "";
            buildNotify = notify;

            ComputeErrorCode error = CL10.BuildProgram(Handle, handleCount, deviceHandles, options, buildNotify, notifyDataPtr);
            ComputeException.ThrowOnError(error);
        }

        public ComputeKernel CreateKernel(string functionName)
        {
            return new ComputeKernel(functionName, this);
        }

        public string GetBuildLog(ComputeDevice device)
        {
            return GetStringInfo<CLProgramHandle, CLDeviceHandle, ComputeProgramBuildInfo>(Handle, device.Handle, ComputeProgramBuildInfo.BuildLog, CL10.GetProgramBuildInfo);
        }

        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseProgram(Handle);
                Handle.Invalidate();
            }
        }
    }
}