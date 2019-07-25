using System;
using System.Runtime.InteropServices;
using System.Security;

namespace Cloo.Bindings
{
    [SuppressUnmanagedCodeSecurity]
    public class CL11 : CL10
    {
        [DllImport(libName, EntryPoint = "clSetEventCallback")]
        public static extern ComputeErrorCode SetEventCallback(CLEventHandle @event,int command_exec_callback_type,ComputeEventCallback pfn_notify,IntPtr user_data);
    }

    public delegate void ComputeEventCallback(CLEventHandle eventHandle, int cmdExecStatusOrErr, IntPtr userData);
}