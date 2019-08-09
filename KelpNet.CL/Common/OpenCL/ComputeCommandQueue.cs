using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace KelpNet.CL.Common
{
    public class ComputeCommandQueue : ComputeObject, IDisposable
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeDevice device;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        internal IList<ComputeEvent> Events;


        public ComputeContext Context { get { return context; } }

        public ComputeDevice Device { get { return device; } }

        public void ReadFromBuffer<T>(ComputeBuffer<T> source, ref T[] destination, bool blocking, IList<ComputeEvent> events) where T : unmanaged
        {
            ReadFromBuffer(source, ref destination, blocking, 0, 0, source.Count, events);
        }

        public void ReadFromBuffer<T>(ComputeBuffer<T> source, ref T[] destination, bool blocking, long sourceOffset, long destinationOffset, long region, IList<ComputeEvent> events) where T : unmanaged
        {
            GCHandle destinationGCHandle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr destinationOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(destination, (int)destinationOffset);

            if (blocking)
            {
                Read(source, blocking, sourceOffset, region, destinationOffsetPtr, events);
                destinationGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = events != null && !events.IsReadOnly;
                IList<ComputeEvent> eventList = userEventsWritable ? events : Events;
                Read(source, blocking, sourceOffset, region, destinationOffsetPtr, eventList);
                ComputeEvent newEvent = (ComputeEvent)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(destinationGCHandle);
            }
        }

        public ComputeCommandQueue(ComputeContext context, ComputeDevice device, ComputeCommandQueueFlags properties)
        {
            handle = CL10.CreateCommandQueue(context.handle, device.handle, properties, out _);

            this.device = device;
            this.context = context;

            Events = new List<ComputeEvent>();

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        public void Execute(ComputeKernel kernel, long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize, ICollection<ComputeEvent> events)
        {
            int eventWaitListSize;
            IntPtr[] eventHandles = ComputeTools.ExtractHandles(events, out eventWaitListSize);
            bool eventsWritable = events != null && !events.IsReadOnly;
            IntPtr[] newEventHandle = eventsWritable ? new IntPtr[1] : null;

            CL10.EnqueueNDRangeKernel(handle, kernel.handle, globalWorkSize.Length, ComputeTools.ConvertArray(globalWorkOffset), ComputeTools.ConvertArray(globalWorkSize), ComputeTools.ConvertArray(localWorkSize), eventWaitListSize, eventHandles, newEventHandle);

            if (eventsWritable)
            {
                events.Add(new ComputeEvent(newEventHandle[0], this));
            }
        }

        public void Finish()
        {
            CL10.Finish(handle);
        }

        public void Read<T>(ComputeBuffer<T> source, bool blocking, long offset, long region, IntPtr destination, ICollection<ComputeEvent> events) where T : unmanaged
        {
            int eventWaitListSize;
            IntPtr[] eventHandles = ComputeTools.ExtractHandles(events, out eventWaitListSize);
            bool eventsWritable = events != null && !events.IsReadOnly;
            IntPtr[] newEventHandle = eventsWritable ? new IntPtr[1] : null;
            CL10.EnqueueReadBuffer(handle, source.handle, blocking, new IntPtr(offset * Unsafe.SizeOf<T>()), new IntPtr(region * Unsafe.SizeOf<T>()), destination, eventWaitListSize, eventHandles, newEventHandle);
            

            if (eventsWritable)
            {
                events.Add(new ComputeEvent(newEventHandle[0], this));
            }
        }

        public void Dispose()
        {
#if DEBUG
            Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
            CL10.ReleaseCommandQueue(handle);
        }
    }
}