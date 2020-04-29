using System;
using System.Runtime.InteropServices;
using System.Security;

namespace KelpNet.CL.Common
{
    [SuppressUnmanagedCodeSecurity]
    public class CL10
    {
        protected const string libName = "OpenCL.dll";

        [DllImport(libName, EntryPoint = "clGetPlatformIDs")]
        public static extern int GetPlatformIDs(int num_entries, [Out, MarshalAs(UnmanagedType.LPArray)] IntPtr[] platforms, out int num_platforms);

        [DllImport(libName, EntryPoint = "clGetPlatformInfo")]
        public static extern int GetPlatformInfo(IntPtr platform, ComputePlatformInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clGetDeviceIDs")]
        public static extern int GetDeviceIDs(IntPtr platform, ComputeDeviceTypes device_type, int num_entries, [Out, MarshalAs(UnmanagedType.LPArray)] IntPtr[] devices, out int num_devices);

        [DllImport(libName, EntryPoint = "clGetDeviceInfo")]
        public static extern int GetDeviceInfo(IntPtr device, ComputeDeviceInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateContext")]
        public static extern IntPtr CreateContext([MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties, int num_devices, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] devices, ComputeContextNotifier pfn_notify, IntPtr user_data, out int errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseContext")]
        public static extern int ReleaseContext(IntPtr context);

        [DllImport(libName, EntryPoint = "clCreateCommandQueue")]
        public static extern IntPtr CreateCommandQueue(IntPtr context, IntPtr device, ComputeCommandQueueFlags properties, out int errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseCommandQueue")]
        public static extern int ReleaseCommandQueue(IntPtr command_queue);

        [DllImport(libName, EntryPoint = "clCreateBuffer")]
        public static extern IntPtr CreateBuffer(IntPtr context, ComputeMemoryFlags flags, IntPtr size, IntPtr host_ptr, out int errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseMemObject")]
        public static extern int ReleaseMemObject(IntPtr memobj);

        [DllImport(libName, EntryPoint = "clGetMemObjectInfo")]
        public static extern int GetMemObjectInfo(IntPtr memobj, ComputeMemoryInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateProgramWithSource")]
        public static extern IntPtr CreateProgramWithSource(IntPtr context, int count, string[] strings, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] lengths, out int errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseProgram")]
        public static extern int ReleaseProgram(IntPtr program);

        [DllImport(libName, EntryPoint = "clBuildProgram")]
        public static extern int BuildProgram(IntPtr program, int num_devices, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] device_list, string options, ComputeProgramBuildNotifier pfn_notify, IntPtr user_data);

        [DllImport(libName, EntryPoint = "clGetProgramBuildInfo")]
        public static extern int GetProgramBuildInfo(IntPtr program, IntPtr device, ComputeProgramBuildInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateKernel")]
        public static extern IntPtr CreateKernel(IntPtr program, string kernel_name, out int errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseKernel")]
        public static extern int ReleaseKernel(IntPtr kernel);

        [DllImport(libName, EntryPoint = "clSetKernelArg")]
        public static extern int SetKernelArg(IntPtr kernel, int arg_index, IntPtr arg_size, IntPtr arg_value);

        [DllImport(libName, EntryPoint = "clGetEventInfo")]
        public static extern int GetEventInfo(IntPtr @event, ComputeEventInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clReleaseEvent")]
        public static extern int ReleaseEvent(IntPtr @event);

        [DllImport(libName, EntryPoint = "clFinish")]
        public static extern int Finish(IntPtr command_queue);

        [DllImport(libName, EntryPoint = "clEnqueueReadBuffer")]
        public static extern int EnqueueReadBuffer(IntPtr command_queue, IntPtr buffer, [MarshalAs(UnmanagedType.Bool)] bool blocking_read, IntPtr offset, IntPtr cb, IntPtr ptr, int num_events_in_wait_list, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] event_wait_list, [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] IntPtr[] new_event);

        [DllImport(libName, EntryPoint = "clEnqueueNDRangeKernel")]
        public static extern int EnqueueNDRangeKernel(IntPtr command_queue, IntPtr kernel, int work_dim, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_offset, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_size, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] local_work_size, int num_events_in_wait_list, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] event_wait_list, [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] IntPtr[] new_event);
    }

    public delegate void ComputeContextNotifier(string errorInfo, IntPtr clDataPtr, IntPtr clDataSize, IntPtr userDataPtr);

    public delegate void ComputeProgramBuildNotifier(IntPtr programHandle, IntPtr notifyDataPtr);
}