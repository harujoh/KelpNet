using System;
using System.Diagnostics;
using Cloo.Bindings;

namespace Cloo
{
    public abstract class ComputeEventBase : ComputeResource
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

        public CLEventHandle Handle
        {
            get;
            protected set;
        }

        public ComputeContext Context { get; protected set; }

        public ComputeCommandType Type { get; protected set; }

        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseEvent(Handle);
                Handle.Invalidate();
            }
        }

        protected void HookNotifier()
        {
            statusNotify = StatusNotify;
            ComputeErrorCode error = CL11.SetEventCallback(Handle, (int)ComputeCommandExecutionStatus.Complete, statusNotify, IntPtr.Zero);
            ComputeException.ThrowOnError(error);
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

        private void StatusNotify(CLEventHandle eventHandle, int cmdExecStatusOrErr, IntPtr userData)
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
    }

    public class ComputeCommandStatusArgs : EventArgs
    {
        public ComputeEventBase Event { get; private set; }

        public ComputeCommandExecutionStatus Status { get; private set; }

        public ComputeCommandStatusArgs(ComputeEventBase ev, ComputeCommandExecutionStatus status)
        {
            Event = ev;
            Status = status;
        }
    }

    public delegate void ComputeCommandStatusChanged(object sender, ComputeCommandStatusArgs args);
}