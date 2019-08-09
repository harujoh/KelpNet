using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace KelpNet.CL.Common
{
    public class ComputeEvent : ComputeObject, IDisposable
    {
        private event ComputeCommandStatusChanged aborted;
        private event ComputeCommandStatusChanged completed;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeCommandStatusArgs status;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeEventCallback statusNotify;

        public event ComputeCommandStatusChanged Aborted
        {
            add
            {
                aborted += value;
                if (status != null && status.Status != ComputeCommandExecutionStatus.Complete)
                {
                    value.Invoke(this, status);
                }
            }
            remove
            {
                aborted -= value;
            }
        }

        public event ComputeCommandStatusChanged Completed
        {
            add
            {
                completed += value;
                if (status != null && status.Status == ComputeCommandExecutionStatus.Complete)
                {
                    value.Invoke(this, status);
                }
            }
            remove
            {
                completed -= value;
            }
        }

        public ComputeContext Context { get; protected set; }

        public ComputeCommandType Type { get; protected set; }

        protected void HookNotifier()
        {
            statusNotify = StatusNotify;
            CL11.SetEventCallback(handle, (int)ComputeCommandExecutionStatus.Complete, statusNotify, IntPtr.Zero);
        }

        protected virtual void OnCompleted(object sender, ComputeCommandStatusArgs evArgs)
        {
#if DEBUG
            Trace.WriteLine("Complete " + Type + " operation of " + this + ".", "Information");
#endif
            if (completed != null)
            {
                completed(sender, evArgs);
            }
        }

        protected virtual void OnAborted(object sender, ComputeCommandStatusArgs evArgs)
        {
#if DEBUG
            Trace.WriteLine("Abort " + Type + " operation of " + this + ".", "Information");
#endif
            if (aborted != null)
            {
                aborted(sender, evArgs);
            }
        }

        private void StatusNotify(IntPtr eventHandle, int cmdExecStatusOrErr, IntPtr userData)
        {
            status = new ComputeCommandStatusArgs(this, (ComputeCommandExecutionStatus)cmdExecStatusOrErr);

            switch (cmdExecStatusOrErr)
            {
                case (int)ComputeCommandExecutionStatus.Complete:
                    OnCompleted(this, status);
                    break;

                default:
                    OnAborted(this, status);
                    break;
            }
        }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private GCHandle gcHandle;

        public ComputeCommandQueue CommandQueue { get; private set; }

        internal ComputeEvent(IntPtr handle, ComputeCommandQueue queue)
        {

            CommandQueue = queue;
            Type = (ComputeCommandType)GetInfo<IntPtr, ComputeEventInfo, int>(handle, ComputeEventInfo.CommandType, CL10.GetEventInfo);
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

        public void Dispose()
        {
            FreeGCHandle();
#if DEBUG
            Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
            CL10.ReleaseKernel(handle);
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

    public class ComputeCommandStatusArgs : EventArgs
    {
        public ComputeEvent Event { get; private set; }

        public ComputeCommandExecutionStatus Status { get; private set; }

        public ComputeCommandStatusArgs(ComputeEvent ev, ComputeCommandExecutionStatus status)
        {
            Event = ev;
            Status = status;
        }
    }

    public delegate void ComputeCommandStatusChanged(object sender, ComputeCommandStatusArgs args);
}