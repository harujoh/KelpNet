using System;
using System.Diagnostics;

namespace Cloo
{
    public class ComputeContextProperty
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContextPropertyName name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly IntPtr value;

        public ComputeContextPropertyName Name { get { return name; } }

        public IntPtr Value { get { return value; } }

        public ComputeContextProperty(ComputeContextPropertyName name, IntPtr value)
        {
            this.name = name;
            this.value = value;
        }
    }
}