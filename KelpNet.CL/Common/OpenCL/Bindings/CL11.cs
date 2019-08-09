using System;
using System.Runtime.InteropServices;
using System.Security;

namespace KelpNet.CL.Common
{
    [SuppressUnmanagedCodeSecurity]
    public class CL11 : CL10
    {
        [DllImport(libName, EntryPoint = "clSetEventCallback")]
        public static extern int SetEventCallback(IntPtr @event, int command_exec_callback_type, ComputeEventCallback pfn_notify, IntPtr user_data);
    }

    public delegate void ComputeEventCallback(IntPtr eventHandle, int cmdExecStatusOrErr, IntPtr userData);
}