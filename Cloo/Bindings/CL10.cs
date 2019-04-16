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
        public extern static ComputeErrorCode GetPlatformIDs(
            Int32 num_entries,
            [Out, MarshalAs(UnmanagedType.LPArray)] CLPlatformHandle[] platforms,
            out Int32 num_platforms);

        [DllImport(libName, EntryPoint = "clGetPlatformInfo")]
        public extern static ComputeErrorCode GetPlatformInfo(
            CLPlatformHandle platform,
            ComputePlatformInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clGetDeviceIDs")]
        public extern static ComputeErrorCode GetDeviceIDs(
            CLPlatformHandle platform,
            ComputeDeviceTypes device_type,
            Int32 num_entries,
            [Out, MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices,
            out Int32 num_devices);

        [DllImport(libName, EntryPoint = "clGetDeviceInfo")]
        public extern static ComputeErrorCode GetDeviceInfo(
            CLDeviceHandle device,
            ComputeDeviceInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateContext")]
        public extern static CLContextHandle CreateContext(
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties,
            Int32 num_devices,
            [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices,
            ComputeContextNotifier pfn_notify,
            IntPtr user_data,
            out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseContext")]
        public extern static ComputeErrorCode ReleaseContext(
            CLContextHandle context);

        [DllImport(libName, EntryPoint = "clGetContextInfo")]
        public extern static ComputeErrorCode GetContextInfo(
            CLContextHandle context,
            ComputeContextInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateCommandQueue")]
        public extern static CLCommandQueueHandle CreateCommandQueue(
            CLContextHandle context,
            CLDeviceHandle device,
            ComputeCommandQueueFlags properties,
            out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseCommandQueue")]
        public extern static ComputeErrorCode
        ReleaseCommandQueue(
            CLCommandQueueHandle command_queue);

        [DllImport(libName, EntryPoint = "clCreateBuffer")]
        public extern static CLMemoryHandle CreateBuffer(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            IntPtr size,
            IntPtr host_ptr,
            out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseMemObject")]
        public extern static ComputeErrorCode ReleaseMemObject(
            CLMemoryHandle memobj);

        [DllImport(libName, EntryPoint = "clGetMemObjectInfo")]
        public extern static ComputeErrorCode GetMemObjectInfo(
            CLMemoryHandle memobj,
            ComputeMemoryInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateProgramWithSource")]
        public extern static CLProgramHandle CreateProgramWithSource(
            CLContextHandle context,
            Int32 count,
            String[] strings,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] lengths,
            out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseProgram")]
        public extern static ComputeErrorCode ReleaseProgram(
            CLProgramHandle program);

        [DllImport(libName, EntryPoint = "clBuildProgram")]
        public extern static ComputeErrorCode BuildProgram(
            CLProgramHandle program,
            Int32 num_devices,
            [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] device_list,
            String options,
            ComputeProgramBuildNotifier pfn_notify,
            IntPtr user_data);

        [DllImport(libName, EntryPoint = "clGetProgramBuildInfo")]
        public extern static ComputeErrorCode GetProgramBuildInfo(
            CLProgramHandle program,
            CLDeviceHandle device,
            ComputeProgramBuildInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clCreateKernel")]
        public extern static CLKernelHandle CreateKernel(
            CLProgramHandle program,
            String kernel_name,
            out ComputeErrorCode errcode_ret);

        [DllImport(libName, EntryPoint = "clReleaseKernel")]
        public extern static ComputeErrorCode ReleaseKernel(
            CLKernelHandle kernel);

        [DllImport(libName, EntryPoint = "clSetKernelArg")]
        public extern static ComputeErrorCode SetKernelArg(
            CLKernelHandle kernel,
            Int32 arg_index,
            IntPtr arg_size,
            IntPtr arg_value);

        [DllImport(libName, EntryPoint = "clGetEventInfo")]
        public extern static ComputeErrorCode GetEventInfo(
            CLEventHandle @event,
            ComputeEventInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        [DllImport(libName, EntryPoint = "clReleaseEvent")]
        public extern static ComputeErrorCode ReleaseEvent(
            CLEventHandle @event);

        [DllImport(libName, EntryPoint = "clFinish")]
        public extern static ComputeErrorCode Finish(
            CLCommandQueueHandle command_queue);

        [DllImport(libName, EntryPoint = "clEnqueueReadBuffer")]
        public extern static ComputeErrorCode EnqueueReadBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_read,
            IntPtr offset,
            IntPtr cb,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);

        [DllImport(libName, EntryPoint = "clEnqueueNDRangeKernel")]
        public extern static ComputeErrorCode EnqueueNDRangeKernel(
            CLCommandQueueHandle command_queue,
            CLKernelHandle kernel,
            Int32 work_dim,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_offset,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_size,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] local_work_size,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);
    }

    public delegate void ComputeContextNotifier(String errorInfo, IntPtr clDataPtr, IntPtr clDataSize, IntPtr userDataPtr);

    public delegate void ComputeProgramBuildNotifier(CLProgramHandle programHandle, IntPtr notifyDataPtr);
}