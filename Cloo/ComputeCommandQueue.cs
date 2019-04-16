using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public partial class ComputeCommandQueue : ComputeResource
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeDevice device;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private bool outOfOrderExec;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private bool profiling;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        internal IList<ComputeEventBase> Events;

        public CLCommandQueueHandle Handle
        {
            get;
            protected set;
        }

        public ComputeContext Context { get { return context; } }

        public ComputeDevice Device { get { return device; } }

        public ComputeCommandQueue(ComputeContext context, ComputeDevice device, ComputeCommandQueueFlags properties)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL11.CreateCommandQueue(context.Handle, device.Handle, properties, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            this.device = device;
            this.context = context;

            outOfOrderExec = ((properties & ComputeCommandQueueFlags.OutOfOrderExecution) == ComputeCommandQueueFlags.OutOfOrderExecution);
            profiling = ((properties & ComputeCommandQueueFlags.Profiling) == ComputeCommandQueueFlags.Profiling);

            Events = new List<ComputeEventBase>();

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void Execute(ComputeKernel kernel, long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize, ICollection<ComputeEventBase> events)
        {
            int eventWaitListSize;
            CLEventHandle[] eventHandles = ComputeTools.ExtractHandles(events, out eventWaitListSize);
            bool eventsWritable = (events != null && !events.IsReadOnly);
            CLEventHandle[] newEventHandle = (eventsWritable) ? new CLEventHandle[1] : null;

            ComputeErrorCode error = CL10.EnqueueNDRangeKernel(Handle, kernel.Handle, globalWorkSize.Length, ComputeTools.ConvertArray(globalWorkOffset), ComputeTools.ConvertArray(globalWorkSize), ComputeTools.ConvertArray(localWorkSize), eventWaitListSize, eventHandles, newEventHandle);
            ComputeException.ThrowOnError(error);

            if (eventsWritable)
                events.Add(new ComputeEvent(newEventHandle[0], this));
        }

        public void Finish()
        {
            ComputeErrorCode error = CL10.Finish(Handle);
            ComputeException.ThrowOnError(error);
        }

        public void Read<T>(ComputeBufferBase<T> source, bool blocking, long offset, long region, IntPtr destination, ICollection<ComputeEventBase> events) where T : struct
        {
            int sizeofT = Marshal.SizeOf(typeof(T));

            int eventWaitListSize;
            CLEventHandle[] eventHandles = ComputeTools.ExtractHandles(events, out eventWaitListSize);
            bool eventsWritable = (events != null && !events.IsReadOnly);
            CLEventHandle[] newEventHandle = (eventsWritable) ? new CLEventHandle[1] : null;

            ComputeErrorCode error = CL10.EnqueueReadBuffer(Handle, source.Handle, blocking, new IntPtr(offset * sizeofT), new IntPtr(region * sizeofT), destination, eventWaitListSize, eventHandles, newEventHandle);
            ComputeException.ThrowOnError(error);

            if (eventsWritable)
                events.Add(new ComputeEvent(newEventHandle[0], this));
        }

        protected override void Dispose(bool manual)
        {
            if (manual)
            {
                //free managed resources
            }

            // free native resources
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseCommandQueue(Handle);
                Handle.Invalidate();
            }
        }
    }
}