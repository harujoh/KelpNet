using System;
using System.Runtime.InteropServices;
using System.Security;

namespace Cloo.Bindings
{
    [SuppressUnmanagedCodeSecurity]
    public class CL10
    {
        protected const string libName = "OpenCL.dll";

        [DllImport(libName, EntryPoint = "clGetPlatformIDs")]
        public static extern ComputeErrorCode GetPlatformIDs(int num_entries, [Out, MarshalAs(UnmanagedType.LPArray)] CLPlatformHandle[] platforms, out int num_platforms);

        [DllImport(libName, EntryPoint = "clGetPlatformInfo")]
        public static extern ComputeErrorCode GetPlatformInfo(CLPlatformHandle platform, ComputePlatformInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clGetDeviceIDs")]
        public static extern ComputeErrorCode GetDeviceIDs(CLPlatformHandle platform, ComputeDeviceTypes device_type, int num_entries, [Out, MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices, out int num_devices);

        [DllImport(libName, EntryPoint = "clGetDeviceInfo")]
        public static extern ComputeErrorCode GetDeviceInfo(CLDeviceHandle device, ComputeDeviceInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateContext")]
        public static extern CLContextHandle CreateContext([MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties, int num_devices, [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices, ComputeContextNotifier pfn_notify, IntPtr user_data, out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseContext")]
        public static extern ComputeErrorCode ReleaseContext(CLContextHandle context);

        [DllImport(libName, EntryPoint = "clCreateCommandQueue")]
        public static extern CLCommandQueueHandle CreateCommandQueue(CLContextHandle context, CLDeviceHandle device, ComputeCommandQueueFlags properties, out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseCommandQueue")]
        public static extern ComputeErrorCode ReleaseCommandQueue(CLCommandQueueHandle command_queue);

        [DllImport(libName, EntryPoint = "clCreateBuffer")]
        public static extern CLMemoryHandle CreateBuffer(CLContextHandle context, ComputeMemoryFlags flags, IntPtr size, IntPtr host_ptr, out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseMemObject")]
        public static extern ComputeErrorCode ReleaseMemObject(CLMemoryHandle memobj);

        [DllImport(libName, EntryPoint = "clGetMemObjectInfo")]
        public static extern ComputeErrorCode GetMemObjectInfo(CLMemoryHandle memobj, ComputeMemoryInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateProgramWithSource")]
        public static extern CLProgramHandle CreateProgramWithSource(CLContextHandle context, int count, string[] strings, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] lengths, out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseProgram")]
        public static extern ComputeErrorCode ReleaseProgram(CLProgramHandle program);

        [DllImport(libName, EntryPoint = "clBuildProgram")]
        public static extern ComputeErrorCode BuildProgram(CLProgramHandle program, int num_devices, [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] device_list, string options, ComputeProgramBuildNotifier pfn_notify, IntPtr user_data);

        [DllImport(libName, EntryPoint = "clGetProgramBuildInfo")]
        public static extern ComputeErrorCode GetProgramBuildInfo(CLProgramHandle program, CLDeviceHandle device, ComputeProgramBuildInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateKernel")]
        public static extern CLKernelHandle CreateKernel(CLProgramHandle program, string kernel_name, out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseKernel")]
        public static extern ComputeErrorCode ReleaseKernel(CLKernelHandle kernel);

        [DllImport(libName, EntryPoint = "clSetKernelArg")]
        public static extern ComputeErrorCode SetKernelArg(CLKernelHandle kernel, int arg_index, IntPtr arg_size, IntPtr arg_value);

        [DllImport(libName, EntryPoint = "clGetEventInfo")]
        public static extern ComputeErrorCode GetEventInfo(CLEventHandle @event, ComputeEventInfo param_name, IntPtr param_value_size, IntPtr param_value, out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clReleaseEvent")]
        public static extern ComputeErrorCode ReleaseEvent(CLEventHandle @event);

        [DllImport(libName, EntryPoint = "clFinish")]
        public static extern ComputeErrorCode Finish(CLCommandQueueHandle command_queue);

        [DllImport(libName, EntryPoint = "clEnqueueReadBuffer")]
        public static extern ComputeErrorCode EnqueueReadBuffer(CLCommandQueueHandle command_queue, CLMemoryHandle buffer, [MarshalAs(UnmanagedType.Bool)] bool blocking_read, IntPtr offset, IntPtr cb, IntPtr ptr, int num_events_in_wait_list, [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list, [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);

        [DllImport(libName, EntryPoint = "clEnqueueNDRangeKernel")]
        public static extern ComputeErrorCode EnqueueNDRangeKernel(CLCommandQueueHandle command_queue, CLKernelHandle kernel, int work_dim, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_offset, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_size, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] local_work_size, int num_events_in_wait_list, [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list, [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);
    }

    public delegate void ComputeContextNotifier(string errorInfo, IntPtr clDataPtr, IntPtr clDataSize, IntPtr userDataPtr);

    public delegate void ComputeProgramBuildNotifier(CLProgramHandle programHandle, IntPtr notifyDataPtr);
}