#region License

/*

Copyright (c) 2009 - 2011 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo.Bindings
{
    using System;
    using System.Diagnostics;
    using System.Runtime.InteropServices;
    using System.Security;

    /// <summary>
    /// Contains bindings to the OpenCL 1.1 functions.
    /// </summary>
    /// <remarks> See the OpenCL specification for documentation regarding these functions. </remarks>
    [SuppressUnmanagedCodeSecurity]
    public class CL11 : CL10
    {
        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateSubBuffer")]
        public extern static CLMemoryHandle CreateSubBuffer(
            CLMemoryHandle buffer,
            ComputeMemoryFlags flags,
            ComputeBufferCreateType buffer_create_type,
            ref SysIntX2 buffer_create_info,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clSetMemObjectDestructorCallback")]
        public extern static ComputeErrorCode SetMemObjectDestructorCallback( 
            CLMemoryHandle memobj, 
            ComputeMemoryDestructorNotifer pfn_notify, 
            IntPtr user_data);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateUserEvent")]
        public extern static CLEventHandle CreateUserEvent(
            CLContextHandle context,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clSetUserEventStatus")]
        public extern static ComputeErrorCode SetUserEventStatus(
            CLEventHandle @event,
            Int32 execution_status);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clSetEventCallback")]
        public extern static ComputeErrorCode SetEventCallback(
            CLEventHandle @event,
            Int32 command_exec_callback_type,
            ComputeEventCallback pfn_notify,
            IntPtr user_data);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueReadBufferRect")]
        public extern static ComputeErrorCode EnqueueReadBufferRect(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_read,
            ref SysIntX3 buffer_offset,
            ref SysIntX3 host_offset,
            ref SysIntX3 region,
            IntPtr buffer_row_pitch,
            IntPtr buffer_slice_pitch,
            IntPtr host_row_pitch,
            IntPtr host_slice_pitch,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueWriteBufferRect")]
        public extern static ComputeErrorCode EnqueueWriteBufferRect(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_write,
            ref SysIntX3 buffer_offset,
            ref SysIntX3 host_offset,
            ref SysIntX3 region,
            IntPtr buffer_row_pitch,
            IntPtr buffer_slice_pitch,
            IntPtr host_row_pitch,
            IntPtr host_slice_pitch,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueCopyBufferRect")]
        public extern static ComputeErrorCode EnqueueCopyBufferRect(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_buffer,
            CLMemoryHandle dst_buffer,
            ref SysIntX3 src_origin,
            ref SysIntX3 dst_origin,
            ref SysIntX3 region,
            IntPtr src_row_pitch,
            IntPtr src_slice_pitch,
            IntPtr dst_row_pitch,
            IntPtr dst_slice_pitch,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [Obsolete("This function has been deprecated in OpenCL 1.1.")]
        new public static ComputeErrorCode SetCommandQueueProperty(
            CLCommandQueueHandle command_queue,
            ComputeCommandQueueFlags properties,
            [MarshalAs(UnmanagedType.Bool)] bool enable,
            out ComputeCommandQueueFlags old_properties)
        {
            Trace.WriteLine("WARNING! clSetCommandQueueProperty has been deprecated in OpenCL 1.1.");
            return CL10.SetCommandQueueProperty(command_queue, properties, enable, out old_properties);
        }
    }

    /// <summary>
    /// A callback function that can be registered by the application.
    /// </summary>
    /// <param name="memobj"> The memory object being deleted. When the user callback is called, this memory object is not longer valid. <paramref name="memobj"/> is only provided for reference purposes. </param>
    /// <param name="user_data"> A pointer to user supplied data. </param>
    /// /// <remarks> This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe. </remarks>
    public delegate void ComputeMemoryDestructorNotifer(CLMemoryHandle memobj, IntPtr user_data);

    /// <summary>
    /// The event callback function that can be registered by the application.
    /// </summary>
    /// <param name="eventHandle"> The event object for which the callback function is invoked. </param>
    /// <param name="cmdExecStatusOrErr"> Represents the execution status of the command for which this callback function is invoked. If the callback is called as the result of the command associated with the event being abnormally terminated, an appropriate error code for the error that caused the termination will be passed to <paramref name="cmdExecStatusOrErr"/> instead. </param>
    /// <param name="userData"> A pointer to user supplied data. </param>
    /// /// <remarks> This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe. </remarks>
    public delegate void ComputeEventCallback(CLEventHandle eventHandle, int cmdExecStatusOrErr, IntPtr userData);
}