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
    using System.Runtime.InteropServices;
    using System.Security;

    /// <summary>
    /// Contains bindings to the OpenCL 1.0 functions.
    /// </summary>
    /// <remarks> See the OpenCL specification for documentation regarding these functions. </remarks>
    [SuppressUnmanagedCodeSecurity]
    public class CL10
    {
        /// <summary>
        /// The name of the library that contains the available OpenCL function points.
        /// </summary>
        protected const string libName = "OpenCL.dll";

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetPlatformIDs")]
        public extern static ComputeErrorCode GetPlatformIDs(
            Int32 num_entries,
            [Out, MarshalAs(UnmanagedType.LPArray)] CLPlatformHandle[] platforms,
            out Int32 num_platforms);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetPlatformInfo")]
        public extern static ComputeErrorCode GetPlatformInfo(
            CLPlatformHandle platform,
            ComputePlatformInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetDeviceIDs")]
        public extern static ComputeErrorCode GetDeviceIDs(
            CLPlatformHandle platform,
            ComputeDeviceTypes device_type,
            Int32 num_entries,
            [Out, MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices,
            out Int32 num_devices);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetDeviceInfo")]
        public extern static ComputeErrorCode GetDeviceInfo(
            CLDeviceHandle device,
            ComputeDeviceInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateContext")]
        public extern static CLContextHandle CreateContext(
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties,
            Int32 num_devices,
            [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] devices,
            ComputeContextNotifier pfn_notify,
            IntPtr user_data,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateContextFromType")]
        public extern static CLContextHandle CreateContextFromType(
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties,
            ComputeDeviceTypes device_type,
            ComputeContextNotifier pfn_notify,
            IntPtr user_data,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainContext")]
        public extern static ComputeErrorCode RetainContext(
            CLContextHandle context);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseContext")]
        public extern static ComputeErrorCode ReleaseContext(
            CLContextHandle context);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetContextInfo")]
        public extern static ComputeErrorCode GetContextInfo(
            CLContextHandle context,
            ComputeContextInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateCommandQueue")]
        public extern static CLCommandQueueHandle CreateCommandQueue(
            CLContextHandle context,
            CLDeviceHandle device,
            ComputeCommandQueueFlags properties,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainCommandQueue")]
        public extern static ComputeErrorCode RetainCommandQueue(
            CLCommandQueueHandle command_queue);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseCommandQueue")]
        public extern static ComputeErrorCode
        ReleaseCommandQueue(
            CLCommandQueueHandle command_queue);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetCommandQueueInfo")]
        public extern static ComputeErrorCode GetCommandQueueInfo(
            CLCommandQueueHandle command_queue,
            ComputeCommandQueueInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clSetCommandQueueProperty")]
        public extern static ComputeErrorCode SetCommandQueueProperty(
            CLCommandQueueHandle command_queue,
            ComputeCommandQueueFlags properties,
            [MarshalAs(UnmanagedType.Bool)] bool enable,
            out ComputeCommandQueueFlags old_properties);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateBuffer")]
        public extern static CLMemoryHandle CreateBuffer(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            IntPtr size,
            IntPtr host_ptr,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateImage2D")]
        public extern static CLMemoryHandle CreateImage2D(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            ref ComputeImageFormat image_format,
            IntPtr image_width,
            IntPtr image_height,
            IntPtr image_row_pitch,
            IntPtr host_ptr,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateImage3D")]
        public extern static CLMemoryHandle CreateImage3D(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            ref ComputeImageFormat image_format,
            IntPtr image_width,
            IntPtr image_height,
            IntPtr image_depth,
            IntPtr image_row_pitch,
            IntPtr image_slice_pitch,
            IntPtr host_ptr,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainMemObject")]
        public extern static ComputeErrorCode RetainMemObject(
            CLMemoryHandle memobj);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseMemObject")]
        public extern static ComputeErrorCode ReleaseMemObject(
            CLMemoryHandle memobj);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetSupportedImageFormats")]
        public extern static ComputeErrorCode GetSupportedImageFormats(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            ComputeMemoryType image_type,
            Int32 num_entries,
            [Out, MarshalAs(UnmanagedType.LPArray)] ComputeImageFormat[] image_formats,
            out Int32 num_image_formats);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetMemObjectInfo")]
        public extern static ComputeErrorCode GetMemObjectInfo(
            CLMemoryHandle memobj,
            ComputeMemoryInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetImageInfo")]
        public extern static ComputeErrorCode GetImageInfo(
            CLMemoryHandle image,
            ComputeImageInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateSampler")]
        public extern static CLSamplerHandle CreateSampler(
            CLContextHandle context,
            [MarshalAs(UnmanagedType.Bool)] bool normalized_coords,
            ComputeImageAddressing addressing_mode,
            ComputeImageFiltering filter_mode,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainSampler")]
        public extern static ComputeErrorCode RetainSampler(
            CLSamplerHandle sample);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseSampler")]
        public extern static ComputeErrorCode ReleaseSampler(
            CLSamplerHandle sample);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetSamplerInfo")]
        public extern static ComputeErrorCode GetSamplerInfo(
            CLSamplerHandle sample,
            ComputeSamplerInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateProgramWithSource")]
        public extern static CLProgramHandle CreateProgramWithSource(
            CLContextHandle context,
            Int32 count,
            String[] strings,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] lengths,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateProgramWithBinary")]
        public extern static CLProgramHandle CreateProgramWithBinary(
            CLContextHandle context,
            Int32 num_devices,
            [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] device_list,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] lengths,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] binaries,
            [MarshalAs(UnmanagedType.LPArray)] Int32[] binary_status,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainProgram")]
        public extern static ComputeErrorCode RetainProgram(
            CLProgramHandle program);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseProgram")]
        public extern static ComputeErrorCode ReleaseProgram(
            CLProgramHandle program);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clBuildProgram")]
        public extern static ComputeErrorCode BuildProgram(
            CLProgramHandle program,
            Int32 num_devices,
            [MarshalAs(UnmanagedType.LPArray)] CLDeviceHandle[] device_list,
            String options,
            ComputeProgramBuildNotifier pfn_notify,
            IntPtr user_data);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clUnloadCompiler")]
        public extern static ComputeErrorCode UnloadCompiler();

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetProgramInfo")]
        public extern static ComputeErrorCode GetProgramInfo(
            CLProgramHandle program,
            ComputeProgramInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetProgramBuildInfo")]
        public extern static ComputeErrorCode GetProgramBuildInfo(
            CLProgramHandle program,
            CLDeviceHandle device,
            ComputeProgramBuildInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateKernel")]
        public extern static CLKernelHandle CreateKernel(
            CLProgramHandle program,
            String kernel_name,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateKernelsInProgram")]
        public extern static ComputeErrorCode CreateKernelsInProgram(
            CLProgramHandle program,
            Int32 num_kernels,
            [Out, MarshalAs(UnmanagedType.LPArray)] CLKernelHandle[] kernels,
            out Int32 num_kernels_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainKernel")]
        public extern static ComputeErrorCode RetainKernel(
            CLKernelHandle kernel);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseKernel")]
        public extern static ComputeErrorCode ReleaseKernel(
            CLKernelHandle kernel);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clSetKernelArg")]
        public extern static ComputeErrorCode SetKernelArg(
            CLKernelHandle kernel,
            Int32 arg_index,
            IntPtr arg_size,
            IntPtr arg_value);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetKernelInfo")]
        public extern static ComputeErrorCode GetKernelInfo(
            CLKernelHandle kernel,
            ComputeKernelInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetKernelWorkGroupInfo")]
        public extern static ComputeErrorCode GetKernelWorkGroupInfo(
            CLKernelHandle kernel,
            CLDeviceHandle device,
            ComputeKernelWorkGroupInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clWaitForEvents")]
        public extern static ComputeErrorCode WaitForEvents(
            Int32 num_events,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_list);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetEventInfo")]
        public extern static ComputeErrorCode GetEventInfo(
            CLEventHandle @event,
            ComputeEventInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clRetainEvent")]
        public extern static ComputeErrorCode RetainEvent(
            CLEventHandle @event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clReleaseEvent")]
        public extern static ComputeErrorCode ReleaseEvent(
            CLEventHandle @event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetEventProfilingInfo")]
        public extern static ComputeErrorCode GetEventProfilingInfo(
            CLEventHandle @event,
            ComputeCommandProfilingInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clFlush")]
        public extern static ComputeErrorCode Flush(
            CLCommandQueueHandle command_queue);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clFinish")]
        public extern static ComputeErrorCode Finish(
            CLCommandQueueHandle command_queue);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
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
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueWriteBuffer")]
        public extern static ComputeErrorCode EnqueueWriteBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_write,
            IntPtr offset,
            IntPtr cb,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueCopyBuffer")]
        public extern static ComputeErrorCode EnqueueCopyBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_buffer,
            CLMemoryHandle dst_buffer,
            IntPtr src_offset,
            IntPtr dst_offset,
            IntPtr cb,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueReadImage")]
        public extern static ComputeErrorCode EnqueueReadImage(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle image,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_read,
            ref SysIntX3 origin,
            ref SysIntX3 region,
            IntPtr row_pitch,
            IntPtr slice_pitch,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueWriteImage")]
        public extern static ComputeErrorCode EnqueueWriteImage(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle image,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_write,
            ref SysIntX3 origin,
            ref SysIntX3 region,
            IntPtr input_row_pitch,
            IntPtr input_slice_pitch,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueCopyImage")]
        public extern static ComputeErrorCode EnqueueCopyImage(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_image,
            CLMemoryHandle dst_image,
            ref SysIntX3 src_origin,
            ref SysIntX3 dst_origin,
            ref SysIntX3 region,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueCopyImageToBuffer")]
        public extern static ComputeErrorCode EnqueueCopyImageToBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_image,
            CLMemoryHandle dst_buffer,
            ref SysIntX3 src_origin,
            ref SysIntX3 region,
            IntPtr dst_offset,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueCopyBufferToImage")]
        public extern static ComputeErrorCode EnqueueCopyBufferToImage(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_buffer,
            CLMemoryHandle dst_image,
            IntPtr src_offset,
            ref SysIntX3 dst_origin,
            ref SysIntX3 region,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueMapBuffer")]
        public extern static IntPtr EnqueueMapBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_map,
            ComputeMemoryMappingFlags map_flags,
            IntPtr offset,
            IntPtr cb,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueMapImage")]
        public extern static IntPtr EnqueueMapImage(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle image,
            [MarshalAs(UnmanagedType.Bool)] bool blocking_map,
            ComputeMemoryMappingFlags map_flags,
            ref SysIntX3 origin,
            ref SysIntX3 region,
            out IntPtr image_row_pitch,
            out IntPtr image_slice_pitch,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueUnmapMemObject")]
        public extern static ComputeErrorCode EnqueueUnmapMemObject(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle memobj,
            IntPtr mapped_ptr,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
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
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueTask")]
        public extern static ComputeErrorCode EnqueueTask(
            CLCommandQueueHandle command_queue,
            CLKernelHandle kernel,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        // <summary>
        // See the OpenCL specification.
        // </summary>
        /*
        [DllImport(libName, EntryPoint = "clEnqueueNativeKernel")]
        public extern static ComputeErrorCode EnqueueNativeKernel(
            CLCommandQueueHandle command_queue,
            IntPtr user_func,
            IntPtr args,
            IntPtr cb_args,
            Int32 num_mem_objects,
            IntPtr* mem_list,
            IntPtr* args_mem_loc,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);
        */

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueMarker")]
        public extern static ComputeErrorCode EnqueueMarker(
            CLCommandQueueHandle command_queue,
            out CLEventHandle new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueWaitForEvents")]
        public extern static ComputeErrorCode EnqueueWaitForEvents(
            CLCommandQueueHandle command_queue,
            Int32 num_events,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_list);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueBarrier")]
        public extern static ComputeErrorCode EnqueueBarrier(
            CLCommandQueueHandle command_queue);

        
        /// <summary>
        /// Gets the extension function address for the given function name,
        /// or NULL if a valid function can not be found. The client must
        /// check to make sure the address is not NULL, before using or 
        /// calling the returned function address.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetExtensionFunctionAddress")]
        public extern static IntPtr GetExtensionFunctionAddress(
            String func_name);

        /**************************************************************************************/
        // CL/GL Sharing API

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateFromGLBuffer")]
        public extern static CLMemoryHandle CreateFromGLBuffer(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            Int32 bufobj,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateFromGLTexture2D")]
        public extern static CLMemoryHandle CreateFromGLTexture2D(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            Int32 target,
            Int32 miplevel,
            Int32 texture,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateFromGLTexture3D")]
        public extern static CLMemoryHandle CreateFromGLTexture3D(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            Int32 target,
            Int32 miplevel,
            Int32 texture,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clCreateFromGLRenderbuffer")]
        public extern static CLMemoryHandle CreateFromGLRenderbuffer(
            CLContextHandle context,
            ComputeMemoryFlags flags,
            Int32 renderbuffer,
            out ComputeErrorCode errcode_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetGLObjectInfo")]
        public extern static ComputeErrorCode GetGLObjectInfo(
            CLMemoryHandle memobj,
            out ComputeGLObjectType gl_object_type,
            out Int32 gl_object_name);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clGetGLTextureInfo")]
        public extern static ComputeErrorCode GetGLTextureInfo(
            CLMemoryHandle memobj,
            ComputeGLTextureInfo param_name,
            IntPtr param_value_size,
            IntPtr param_value,
            out IntPtr param_value_size_ret);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueAcquireGLObjects")]
        public extern static ComputeErrorCode EnqueueAcquireGLObjects(
            CLCommandQueueHandle command_queue,
            Int32 num_objects,
            [MarshalAs(UnmanagedType.LPArray)] CLMemoryHandle[] mem_objects,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);

        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueReleaseGLObjects")]
        public extern static ComputeErrorCode EnqueueReleaseGLObjects(
            CLCommandQueueHandle command_queue,
            Int32 num_objects,
            [MarshalAs(UnmanagedType.LPArray)] CLMemoryHandle[] mem_objects,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst=1)] CLEventHandle[] new_event);
    }

    /// <summary>
    /// A callback function that can be registered by the application to report information on errors that occur in the <see cref="ComputeContext"/>.
    /// </summary>
    /// <param name="errorInfo"> An error string. </param>
    /// <param name="clDataPtr"> A pointer to binary data that is returned by the OpenCL implementation that can be used to log additional information helpful in debugging the error.</param>
    /// <param name="clDataSize"> The size of the binary data that is returned by the OpenCL. </param>
    /// <param name="userDataPtr"> The pointer to the optional user data specified in <paramref name="userDataPtr"/> argument of <see cref="ComputeContext"/> constructor. </param>
    /// <remarks> This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe. </remarks>
    public delegate void ComputeContextNotifier(String errorInfo, IntPtr clDataPtr, IntPtr clDataSize, IntPtr userDataPtr);

    /// <summary>
    /// A callback function that can be registered by the application to report the <see cref="ComputeProgram"/> build status.
    /// </summary>
    /// <param name="programHandle"> The handle of the <see cref="ComputeProgram"/> being built. </param>
    /// <param name="notifyDataPtr"> The pointer to the optional user data specified in <paramref name="notifyDataPtr"/> argument of <see cref="ComputeProgram.Build"/>. </param>
    /// <remarks> This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe. </remarks>
    public delegate void ComputeProgramBuildNotifier(CLProgramHandle programHandle, IntPtr notifyDataPtr);
}