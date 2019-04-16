using System;
using System.Diagnostics;

namespace Cloo.Bindings
{
    public struct CLDeviceHandle
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        IntPtr value;

        public bool IsValid
        {
            get { return value != IntPtr.Zero; }
        }

        public IntPtr Value
        {
            get { return value; }
        }

        public void Invalidate()
        {
            value = IntPtr.Zero;
        }
    }
}