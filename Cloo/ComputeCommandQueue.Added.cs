using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Cloo
{
    public partial class ComputeCommandQueue
    {
        public void ReadFromBuffer<T>(ComputeBufferBase<T> source, ref T[] destination, bool blocking, IList<ComputeEventBase> events) where T : struct
        {
            ReadFromBuffer(source, ref destination, blocking, 0, 0, source.Count, events);
        }

        public void ReadFromBuffer<T>(ComputeBufferBase<T> source, ref T[] destination, bool blocking, long sourceOffset, long destinationOffset, long region, IList<ComputeEventBase> events) where T : struct
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
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<ComputeEventBase> eventList = (userEventsWritable) ? events : Events;
                Read(source, blocking, sourceOffset, region, destinationOffsetPtr, eventList);
                ComputeEvent newEvent = (ComputeEvent)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(destinationGCHandle);
            }
        }
    }
}