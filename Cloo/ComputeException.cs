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

namespace Cloo
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// Represents an error state that occurred while executing an OpenCL API call.
    /// </summary>
    /// <seealso cref="ComputeErrorCode"/>
    public class ComputeException : ApplicationException
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeErrorCode code;

        #endregion

        #region Properties

        /// <summary>
        /// Gets the <see cref="ComputeErrorCode"/> of the <see cref="ComputeException"/>.
        /// </summary>
        public ComputeErrorCode ComputeErrorCode { get { return code; } }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new <see cref="ComputeException"/> with a specified <see cref="ComputeErrorCode"/>.
        /// </summary>
        /// <param name="code"> A <see cref="ComputeErrorCode"/>. </param>
        public ComputeException(ComputeErrorCode code)
            : base("OpenCL error code detected: " + code.ToString() + ".")
        {
            this.code = code;
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ComputeException"/> if such is encountered.
        /// </summary>
        /// <param name="errorCode"> The value to be checked for an OpenCL error. </param>
        public static void ThrowOnError(int errorCode)
        {
            ThrowOnError((ComputeErrorCode)errorCode);
        }

        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ComputeException"/> if such is encountered.
        /// </summary>
        /// <param name="errorCode"> The OpenCL error code. </param>
        public static void ThrowOnError(ComputeErrorCode errorCode)
        {
            switch (errorCode)
            {
                case ComputeErrorCode.Success:
                    return;

                case ComputeErrorCode.DeviceNotFound:
                    throw new DeviceNotFoundComputeException();

                case ComputeErrorCode.DeviceNotAvailable:
                    throw new DeviceNotAvailableComputeException();

                case ComputeErrorCode.CompilerNotAvailable:
                    throw new CompilerNotAvailableComputeException();

                case ComputeErrorCode.MemoryObjectAllocationFailure:
                    throw new MemoryObjectAllocationFailureComputeException();

                case ComputeErrorCode.OutOfResources:
                    throw new OutOfResourcesComputeException();

                case ComputeErrorCode.OutOfHostMemory:
                    throw new OutOfHostMemoryComputeException();

                case ComputeErrorCode.ProfilingInfoNotAvailable:
                    throw new ProfilingInfoNotAvailableComputeException();

                case ComputeErrorCode.MemoryCopyOverlap:
                    throw new MemoryCopyOverlapComputeException();

                case ComputeErrorCode.ImageFormatMismatch:
                    throw new ImageFormatMismatchComputeException();

                case ComputeErrorCode.ImageFormatNotSupported:
                    throw new ImageFormatNotSupportedComputeException();

                case ComputeErrorCode.BuildProgramFailure:
                    throw new BuildProgramFailureComputeException();

                case ComputeErrorCode.MapFailure:
                    throw new MapFailureComputeException();

                case ComputeErrorCode.InvalidValue:
                    throw new InvalidValueComputeException();

                case ComputeErrorCode.InvalidDeviceType:
                    throw new InvalidDeviceTypeComputeException();

                case ComputeErrorCode.InvalidPlatform:
                    throw new InvalidPlatformComputeException();

                case ComputeErrorCode.InvalidDevice:
                    throw new InvalidDeviceComputeException();

                case ComputeErrorCode.InvalidContext:
                    throw new InvalidContextComputeException();

                case ComputeErrorCode.InvalidCommandQueueFlags:
                    throw new InvalidCommandQueueFlagsComputeException();

                case ComputeErrorCode.InvalidCommandQueue:
                    throw new InvalidCommandQueueComputeException();

                case ComputeErrorCode.InvalidHostPointer:
                    throw new InvalidHostPointerComputeException();

                case ComputeErrorCode.InvalidMemoryObject:
                    throw new InvalidMemoryObjectComputeException();

                case ComputeErrorCode.InvalidImageFormatDescriptor:
                    throw new InvalidImageFormatDescriptorComputeException();

                case ComputeErrorCode.InvalidImageSize:
                    throw new InvalidImageSizeComputeException();

                case ComputeErrorCode.InvalidSampler:
                    throw new InvalidSamplerComputeException();

                case ComputeErrorCode.InvalidBinary:
                    throw new InvalidBinaryComputeException();

                case ComputeErrorCode.InvalidBuildOptions:
                    throw new InvalidBuildOptionsComputeException();

                case ComputeErrorCode.InvalidProgram:
                    throw new InvalidProgramComputeException();

                case ComputeErrorCode.InvalidProgramExecutable:
                    throw new InvalidProgramExecutableComputeException();

                case ComputeErrorCode.InvalidKernelName:
                    throw new InvalidKernelNameComputeException();

                case ComputeErrorCode.InvalidKernelDefinition:
                    throw new InvalidKernelDefinitionComputeException();

                case ComputeErrorCode.InvalidKernel:
                    throw new InvalidKernelComputeException();

                case ComputeErrorCode.InvalidArgumentIndex:
                    throw new InvalidArgumentIndexComputeException();

                case ComputeErrorCode.InvalidArgumentValue:
                    throw new InvalidArgumentValueComputeException();

                case ComputeErrorCode.InvalidArgumentSize:
                    throw new InvalidArgumentSizeComputeException();

                case ComputeErrorCode.InvalidKernelArguments:
                    throw new InvalidKernelArgumentsComputeException();

                case ComputeErrorCode.InvalidWorkDimension:
                    throw new InvalidWorkDimensionsComputeException();

                case ComputeErrorCode.InvalidWorkGroupSize:
                    throw new InvalidWorkGroupSizeComputeException();

                case ComputeErrorCode.InvalidWorkItemSize:
                    throw new InvalidWorkItemSizeComputeException();

                case ComputeErrorCode.InvalidGlobalOffset:
                    throw new InvalidGlobalOffsetComputeException();

                case ComputeErrorCode.InvalidEventWaitList:
                    throw new InvalidEventWaitListComputeException();

                case ComputeErrorCode.InvalidEvent:
                    throw new InvalidEventComputeException();

                case ComputeErrorCode.InvalidOperation:
                    throw new InvalidOperationComputeException();

                case ComputeErrorCode.InvalidGLObject:
                    throw new InvalidGLObjectComputeException();

                case ComputeErrorCode.InvalidBufferSize:
                    throw new InvalidBufferSizeComputeException();

                case ComputeErrorCode.InvalidMipLevel:
                    throw new InvalidMipLevelComputeException();

                default:
                    throw new ComputeException(errorCode);
            }
        }

        #endregion
    }

    #region Exception classes

    // Disable CS1591 warnings (missing XML comment for publicly visible type or member).
    #pragma warning disable 1591

    public class DeviceNotFoundComputeException : ComputeException
    { public DeviceNotFoundComputeException() : base(ComputeErrorCode.DeviceNotFound) { } }

    public class DeviceNotAvailableComputeException : ComputeException
    { public DeviceNotAvailableComputeException() : base(ComputeErrorCode.DeviceNotAvailable) { } }

    public class CompilerNotAvailableComputeException : ComputeException
    { public CompilerNotAvailableComputeException() : base(ComputeErrorCode.CompilerNotAvailable) { } }

    public class MemoryObjectAllocationFailureComputeException : ComputeException
    { public MemoryObjectAllocationFailureComputeException() : base(ComputeErrorCode.MemoryObjectAllocationFailure) { } }

    public class OutOfResourcesComputeException : ComputeException
    { public OutOfResourcesComputeException() : base(ComputeErrorCode.OutOfResources) { } }

    public class OutOfHostMemoryComputeException : ComputeException
    { public OutOfHostMemoryComputeException() : base(ComputeErrorCode.OutOfHostMemory) { } }

    public class ProfilingInfoNotAvailableComputeException : ComputeException
    { public ProfilingInfoNotAvailableComputeException() : base(ComputeErrorCode.ProfilingInfoNotAvailable) { } }

    public class MemoryCopyOverlapComputeException : ComputeException
    { public MemoryCopyOverlapComputeException() : base(ComputeErrorCode.MemoryCopyOverlap) { } }

    public class ImageFormatMismatchComputeException : ComputeException
    { public ImageFormatMismatchComputeException() : base(ComputeErrorCode.ImageFormatMismatch) { } }

    public class ImageFormatNotSupportedComputeException : ComputeException
    { public ImageFormatNotSupportedComputeException() : base(ComputeErrorCode.ImageFormatNotSupported) { } }

    public class BuildProgramFailureComputeException : ComputeException
    { public BuildProgramFailureComputeException() : base(ComputeErrorCode.BuildProgramFailure) { } }

    public class MapFailureComputeException : ComputeException
    { public MapFailureComputeException() : base(ComputeErrorCode.MapFailure) { } }

    public class InvalidValueComputeException : ComputeException
    { public InvalidValueComputeException() : base(ComputeErrorCode.InvalidValue) { } }

    public class InvalidDeviceTypeComputeException : ComputeException
    { public InvalidDeviceTypeComputeException() : base(ComputeErrorCode.InvalidDeviceType) { } }

    public class InvalidPlatformComputeException : ComputeException
    { public InvalidPlatformComputeException() : base(ComputeErrorCode.InvalidPlatform) { } }

    public class InvalidDeviceComputeException : ComputeException
    { public InvalidDeviceComputeException() : base(ComputeErrorCode.InvalidDevice) { } }

    public class InvalidContextComputeException : ComputeException
    { public InvalidContextComputeException() : base(ComputeErrorCode.InvalidContext) { } }

    public class InvalidCommandQueueFlagsComputeException : ComputeException
    { public InvalidCommandQueueFlagsComputeException() : base(ComputeErrorCode.InvalidCommandQueueFlags) { } }

    public class InvalidCommandQueueComputeException : ComputeException
    { public InvalidCommandQueueComputeException() : base(ComputeErrorCode.InvalidCommandQueue) { } }

    public class InvalidHostPointerComputeException : ComputeException
    { public InvalidHostPointerComputeException() : base(ComputeErrorCode.InvalidHostPointer) { } }

    public class InvalidMemoryObjectComputeException : ComputeException
    { public InvalidMemoryObjectComputeException() : base(ComputeErrorCode.InvalidMemoryObject) { } }

    public class InvalidImageFormatDescriptorComputeException : ComputeException
    { public InvalidImageFormatDescriptorComputeException() : base(ComputeErrorCode.InvalidImageFormatDescriptor) { } }

    public class InvalidImageSizeComputeException : ComputeException
    { public InvalidImageSizeComputeException() : base(ComputeErrorCode.InvalidImageSize) { } }

    public class InvalidSamplerComputeException : ComputeException
    { public InvalidSamplerComputeException() : base(ComputeErrorCode.InvalidSampler) { } }

    public class InvalidBinaryComputeException : ComputeException
    { public InvalidBinaryComputeException() : base(ComputeErrorCode.InvalidBinary) { } }

    public class InvalidBuildOptionsComputeException : ComputeException
    { public InvalidBuildOptionsComputeException() : base(ComputeErrorCode.InvalidBuildOptions) { } }

    public class InvalidProgramComputeException : ComputeException
    { public InvalidProgramComputeException() : base(ComputeErrorCode.InvalidProgram) { } }

    public class InvalidProgramExecutableComputeException : ComputeException
    { public InvalidProgramExecutableComputeException() : base(ComputeErrorCode.InvalidProgramExecutable) { } }

    public class InvalidKernelNameComputeException : ComputeException
    { public InvalidKernelNameComputeException() : base(ComputeErrorCode.InvalidKernelName) { } }

    public class InvalidKernelDefinitionComputeException : ComputeException
    { public InvalidKernelDefinitionComputeException() : base(ComputeErrorCode.InvalidKernelDefinition) { } }

    public class InvalidKernelComputeException : ComputeException
    { public InvalidKernelComputeException() : base(ComputeErrorCode.InvalidKernel) { } }

    public class InvalidArgumentIndexComputeException : ComputeException
    { public InvalidArgumentIndexComputeException() : base(ComputeErrorCode.InvalidArgumentIndex) { } }

    public class InvalidArgumentValueComputeException : ComputeException
    { public InvalidArgumentValueComputeException() : base(ComputeErrorCode.InvalidArgumentValue) { } }

    public class InvalidArgumentSizeComputeException : ComputeException
    { public InvalidArgumentSizeComputeException() : base(ComputeErrorCode.InvalidArgumentSize) { } }

    public class InvalidKernelArgumentsComputeException : ComputeException
    { public InvalidKernelArgumentsComputeException() : base(ComputeErrorCode.InvalidKernelArguments) { } }

    public class InvalidWorkDimensionsComputeException : ComputeException
    { public InvalidWorkDimensionsComputeException() : base(ComputeErrorCode.InvalidWorkDimension) { } }

    public class InvalidWorkGroupSizeComputeException : ComputeException
    { public InvalidWorkGroupSizeComputeException() : base(ComputeErrorCode.InvalidWorkGroupSize) { } }

    public class InvalidWorkItemSizeComputeException : ComputeException
    { public InvalidWorkItemSizeComputeException() : base(ComputeErrorCode.InvalidWorkItemSize) { } }

    public class InvalidGlobalOffsetComputeException : ComputeException
    { public InvalidGlobalOffsetComputeException() : base(ComputeErrorCode.InvalidGlobalOffset) { } }

    public class InvalidEventWaitListComputeException : ComputeException
    { public InvalidEventWaitListComputeException() : base(ComputeErrorCode.InvalidEventWaitList) { } }

    public class InvalidEventComputeException : ComputeException
    { public InvalidEventComputeException() : base(ComputeErrorCode.InvalidEvent) { } }

    public class InvalidOperationComputeException : ComputeException
    { public InvalidOperationComputeException() : base(ComputeErrorCode.InvalidOperation) { } }

    public class InvalidGLObjectComputeException : ComputeException
    { public InvalidGLObjectComputeException() : base(ComputeErrorCode.InvalidGLObject) { } }

    public class InvalidBufferSizeComputeException : ComputeException
    { public InvalidBufferSizeComputeException() : base(ComputeErrorCode.InvalidBufferSize) { } }

    public class InvalidMipLevelComputeException : ComputeException
    { public InvalidMipLevelComputeException() : base(ComputeErrorCode.InvalidMipLevel) { } }

    #endregion
}