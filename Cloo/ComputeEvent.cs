using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeEvent : ComputeEventBase
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private GCHandle gcHandle;

        public ComputeCommandQueue CommandQueue { get; private set; }

        internal ComputeEvent(CLEventHandle handle, ComputeCommandQueue queue)
        {
            Handle = handle;
            SetID(Handle.Value);

            CommandQueue = queue;
            Type = (ComputeCommandType)GetInfo<CLEventHandle, ComputeEventInfo, int>(Handle, ComputeEventInfo.CommandType, CL10.GetEventInfo);
            Context = queue.Context;

            if (ComputeTools.ParseVersionString(CommandQueue.Device.Platform.Version, 1) > new Version(1, 0))
            {
                HookNotifier();
            }

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        internal void TrackGCHandle(GCHandle handle)
        {
            gcHandle = handle;

            Completed += Cleanup;
            Aborted += Cleanup;
        }

        protected override void Dispose(bool manual)
        {
            FreeGCHandle();
            base.Dispose(manual);
        }

        private void Cleanup(object sender, ComputeCommandStatusArgs e)
        {
            lock (CommandQueue.Events)
            {
                if (CommandQueue.Events.Contains(this))
                {
                    CommandQueue.Events.Remove(this);
                    Dispose();
                }
                else
                {
                    FreeGCHandle();
                }
            }
        }

        private void FreeGCHandle()
        {
            if (gcHandle.IsAllocated && gcHandle.Target != null)
            {
                gcHandle.Free();
            }
        }
    }
}