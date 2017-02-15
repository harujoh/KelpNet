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

    /// <summary>
    /// The OpenCL error codes.
    /// </summary>
    public enum ComputeErrorCode : int
    {
        /// <summary> </summary>
        Success = 0,
        /// <summary> </summary>
        DeviceNotFound = -1,
        /// <summary> </summary>
        DeviceNotAvailable = -2,
        /// <summary> </summary>
        CompilerNotAvailable = -3,
        /// <summary> </summary>
        MemoryObjectAllocationFailure = -4,
        /// <summary> </summary>
        OutOfResources = -5,
        /// <summary> </summary>
        OutOfHostMemory = -6,
        /// <summary> </summary>
        ProfilingInfoNotAvailable = -7,
        /// <summary> </summary>
        MemoryCopyOverlap = -8,
        /// <summary> </summary>
        ImageFormatMismatch = -9,
        /// <summary> </summary>
        ImageFormatNotSupported = -10,
        /// <summary> </summary>
        BuildProgramFailure = -11,
        /// <summary> </summary>
        MapFailure = -12,
        /// <summary> </summary>
        MisalignedSubBufferOffset = -13,
        /// <summary> </summary>
        ExecutionStatusErrorForEventsInWaitList = -14,
        /// <summary> </summary>
        InvalidValue = -30,
        /// <summary> </summary>
        InvalidDeviceType = -31,
        /// <summary> </summary>
        InvalidPlatform = -32,
        /// <summary> </summary>
        InvalidDevice = -33,
        /// <summary> </summary>
        InvalidContext = -34,
        /// <summary> </summary>
        InvalidCommandQueueFlags = -35,
        /// <summary> </summary>
        InvalidCommandQueue = -36,
        /// <summary> </summary>
        InvalidHostPointer = -37,
        /// <summary> </summary>
        InvalidMemoryObject = -38,
        /// <summary> </summary>
        InvalidImageFormatDescriptor = -39,
        /// <summary> </summary>
        InvalidImageSize = -40,
        /// <summary> </summary>
        InvalidSampler = -41,
        /// <summary> </summary>
        InvalidBinary = -42,
        /// <summary> </summary>
        InvalidBuildOptions = -43,
        /// <summary> </summary>
        InvalidProgram = -44,
        /// <summary> </summary>
        InvalidProgramExecutable = -45,
        /// <summary> </summary>
        InvalidKernelName = -46,
        /// <summary> </summary>
        InvalidKernelDefinition = -47,
        /// <summary> </summary>
        InvalidKernel = -48,
        /// <summary> </summary>
        InvalidArgumentIndex = -49,
        /// <summary> </summary>
        InvalidArgumentValue = -50,
        /// <summary> </summary>
        InvalidArgumentSize = -51,
        /// <summary> </summary>
        InvalidKernelArguments = -52,
        /// <summary> </summary>
        InvalidWorkDimension = -53,
        /// <summary> </summary>
        InvalidWorkGroupSize = -54,
        /// <summary> </summary>
        InvalidWorkItemSize = -55,
        /// <summary> </summary>
        InvalidGlobalOffset = -56,
        /// <summary> </summary>
        InvalidEventWaitList = -57,
        /// <summary> </summary>
        InvalidEvent = -58,
        /// <summary> </summary>
        InvalidOperation = -59,
        /// <summary> </summary>
        InvalidGLObject = -60,
        /// <summary> </summary>
        InvalidBufferSize = -61,
        /// <summary> </summary>
        InvalidMipLevel = -62,
        /// <summary> </summary>
        InvalidGlobalWorkSize = -63,
        /// <summary> </summary>
        CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR = -1000,
        /// <summary> </summary>
        CL_PLATFORM_NOT_FOUND_KHR = -1001,
        /// <summary> </summary>
        CL_DEVICE_PARTITION_FAILED_EXT = -1057,
        /// <summary> </summary>
        CL_INVALID_PARTITION_COUNT_EXT = -1058,
        /// <summary> </summary>
        CL_INVALID_PARTITION_NAME_EXT = -1059,
    }

    /// <summary>
    /// The OpenCL version.
    /// </summary>
    public enum OpenCLVersion : int
    {
        /// <summary> </summary>
        Version_1_0 = 1,
        /// <summary> </summary>
        Version_1_1 = 1
    }

    /// <summary>
    /// The OpenCL boolean values expressed as integers.
    /// </summary>
    public enum ComputeBoolean : int
    {
        /// <summary> </summary>
        False = 0,
        /// <summary> </summary>
        True = 1
    }

    /// <summary>
    /// The platform info query symbols.
    /// </summary>
    public enum ComputePlatformInfo : int
    {
        /// <summary> </summary>
        Profile = 0x0900,
        /// <summary> </summary>
        Version = 0x0901,
        /// <summary> </summary>
        Name = 0x0902,
        /// <summary> </summary>
        Vendor = 0x0903,
        /// <summary> </summary>
        Extensions = 0x0904,
        /// <summary> </summary>
        CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920,
    }

    /// <summary>
    /// The types of devices.
    /// </summary>
    [Flags]
    public enum ComputeDeviceTypes : long
    {
        /// <summary> </summary>
        Default = 1 << 0,
        /// <summary> </summary>
        Cpu = 1 << 1,
        /// <summary> </summary>
        Gpu = 1 << 2,
        /// <summary> </summary>
        Accelerator = 1 << 3,
        /// <summary> </summary>
        All = 0xFFFFFFFF
    }

    /// <summary>
    /// The device info query symbols.
    /// </summary>
    public enum ComputeDeviceInfo : int
    {
        /// <summary> </summary>
        Type = 0x1000,
        /// <summary> </summary>
        VendorId = 0x1001,
        /// <summary> </summary>
        MaxComputeUnits = 0x1002,
        /// <summary> </summary>
        MaxWorkItemDimensions = 0x1003,
        /// <summary> </summary>
        MaxWorkGroupSize = 0x1004,
        /// <summary> </summary>
        MaxWorkItemSizes = 0x1005,
        /// <summary> </summary>
        PreferredVectorWidthChar = 0x1006,
        /// <summary> </summary>
        PreferredVectorWidthShort = 0x1007,
        /// <summary> </summary>
        PreferredVectorWidthInt = 0x1008,
        /// <summary> </summary>
        PreferredVectorWidthLong = 0x1009,
        /// <summary> </summary>
        PreferredVectorWidthFloat = 0x100A,
        /// <summary> </summary>
        PreferredVectorWidthDouble = 0x100B,
        /// <summary> </summary>
        MaxClockFrequency = 0x100C,
        /// <summary> </summary>
        AddressBits = 0x100D,
        /// <summary> </summary>
        MaxReadImageArguments = 0x100E,
        /// <summary> </summary>
        MaxWriteImageArguments = 0x100F,
        /// <summary> </summary>
        MaxMemoryAllocationSize = 0x1010,
        /// <summary> </summary>
        Image2DMaxWidth = 0x1011,
        /// <summary> </summary>
        Image2DMaxHeight = 0x1012,
        /// <summary> </summary>
        Image3DMaxWidth = 0x1013,
        /// <summary> </summary>
        Image3DMaxHeight = 0x1014,
        /// <summary> </summary>
        Image3DMaxDepth = 0x1015,
        /// <summary> </summary>
        ImageSupport = 0x1016,
        /// <summary> </summary>
        MaxParameterSize = 0x1017,
        /// <summary> </summary>
        MaxSamplers = 0x1018,
        /// <summary> </summary>
        MemoryBaseAddressAlignment = 0x1019,
        /// <summary> </summary>
        MinDataTypeAlignmentSize = 0x101A,
        /// <summary> </summary>
        SingleFPConfig = 0x101B,
        /// <summary> </summary>
        GlobalMemoryCacheType = 0x101C,
        /// <summary> </summary>
        GlobalMemoryCachelineSize = 0x101D,
        /// <summary> </summary>
        GlobalMemoryCacheSize = 0x101E,
        /// <summary> </summary>
        GlobalMemorySize = 0x101F,
        /// <summary> </summary>
        MaxConstantBufferSize = 0x1020,
        /// <summary> </summary>
        MaxConstantArguments = 0x1021,
        /// <summary> </summary>
        LocalMemoryType = 0x1022,
        /// <summary> </summary>
        LocalMemorySize = 0x1023,
        /// <summary> </summary>
        ErrorCorrectionSupport = 0x1024,
        /// <summary> </summary>
        ProfilingTimerResolution = 0x1025,
        /// <summary> </summary>
        EndianLittle = 0x1026,
        /// <summary> </summary>
        Available = 0x1027,
        /// <summary> </summary>
        CompilerAvailable = 0x1028,
        /// <summary> </summary>
        ExecutionCapabilities = 0x1029,
        /// <summary> </summary>
        CommandQueueProperties = 0x102A,
        /// <summary> </summary>
        Name = 0x102B,
        /// <summary> </summary>
        Vendor = 0x102C,
        /// <summary> </summary>
        DriverVersion = 0x102D,
        /// <summary> </summary>
        Profile = 0x102E,
        /// <summary> </summary>
        Version = 0x102F,
        /// <summary> </summary>
        Extensions = 0x1030,
        /// <summary> </summary>
        Platform = 0x1031,
        /// <summary> </summary>
        CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032,
        /// <summary> </summary>
        CL_DEVICE_HALF_FP_CONFIG = 0x1033,
        /// <summary> </summary>
        PreferredVectorWidthHalf = 0x1034,
        /// <summary> </summary>
        HostUnifiedMemory = 0x1035,
        /// <summary> </summary>
        NativeVectorWidthChar = 0x1036,
        /// <summary> </summary>
        NativeVectorWidthShort = 0x1037,
        /// <summary> </summary>
        NativeVectorWidthInt = 0x1038,
        /// <summary> </summary>
        NativeVectorWidthLong = 0x1039,
        /// <summary> </summary>
        NativeVectorWidthFloat = 0x103A,
        /// <summary> </summary>
        NativeVectorWidthDouble = 0x103B,
        /// <summary> </summary>
        NativeVectorWidthHalf = 0x103C,
        /// <summary> </summary>
        OpenCLCVersion = 0x103D,
        /// <summary> </summary>
        CL_DEVICE_PARENT_DEVICE_EXT = 0x4054,
        /// <summary> </summary>
        CL_DEVICE_PARITION_TYPES_EXT = 0x4055,
        /// <summary> </summary>
        CL_DEVICE_AFFINITY_DOMAINS_EXT = 0x4056,
        /// <summary> </summary>
        CL_DEVICE_REFERENCE_COUNT_EXT = 0x4057,
        /// <summary> </summary>
        CL_DEVICE_PARTITION_STYLE_EXT = 0x4058
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum ComputeDeviceSingleCapabilities : long
    {
        /// <summary> </summary>
        Denorm = 1 << 0,
        /// <summary> </summary>
        InfNan = 1 << 1,
        /// <summary> </summary>
        RoundToNearest = 1 << 2,
        /// <summary> </summary>
        RoundToZero = 1 << 3,
        /// <summary> </summary>
        RoundToInf = 1 << 4,
        /// <summary> </summary>
        Fma = 1 << 5,
        /// <summary> </summary>
        SoftFloat = 1 << 6
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeDeviceMemoryCacheType : int
    {
        /// <summary> </summary>
        None = 0x0,
        /// <summary> </summary>
        ReadOnlyCache = 0x1,
        /// <summary> </summary>
        ReadWriteCache = 0x2,
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeDeviceLocalMemoryType : int
    {
        /// <summary> </summary>
        Local = 0x1,
        /// <summary> </summary>
        Global = 0x2
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeDeviceExecutionCapabilities : int
    {
        /// <summary> </summary>
        OpenCLKernel = 1 << 0,
        /// <summary> </summary>
        NativeKernel = 1 << 1
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum ComputeCommandQueueFlags : long
    {
        /// <summary> </summary>
        None = 0,
        /// <summary> </summary>
        OutOfOrderExecution = 1 << 0,
        /// <summary> </summary>
        Profiling = 1 << 1
    }

    /// <summary>
    /// The context info query symbols.
    /// </summary>
    public enum ComputeContextInfo : int
    {
        /// <summary> </summary>
        ReferenceCount = 0x1080,
        /// <summary> </summary>
        Devices = 0x1081,
        /// <summary> </summary>
        Properties = 0x1082,
        /// <summary> </summary>
        NumDevices = 0x1083,
        /// <summary> </summary>
        Platform = 0x1084,
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeContextPropertyName : int
    {
        /// <summary> </summary>
        Platform = ComputeContextInfo.Platform,
        /// <summary> </summary>
        CL_GL_CONTEXT_KHR = 0x2008,
        /// <summary> </summary>
        CL_EGL_DISPLAY_KHR = 0x2009,
        /// <summary> </summary>
        CL_GLX_DISPLAY_KHR = 0x200A,
        /// <summary> </summary>
        CL_WGL_HDC_KHR = 0x200B,
        /// <summary> </summary>
        CL_CGL_SHAREGROUP_KHR = 0x200C,
    }

    /// <summary>
    /// The command queue info query symbols.
    /// </summary>
    public enum ComputeCommandQueueInfo : int
    {
        /// <summary> </summary>
        Context = 0x1090,
        /// <summary> </summary>
        Device = 0x1091,
        /// <summary> </summary>
        ReferenceCount = 0x1092,
        /// <summary> </summary>
        Properties = 0x1093
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum ComputeMemoryFlags : long
    {
        /// <summary> Let the OpenCL choose the default flags. </summary>
        None = 0,
        /// <summary> The <see cref="ComputeMemory"/> will be accessible from the <see cref="ComputeKernel"/> for read and write operations. </summary>
        ReadWrite = 1 << 0,
        /// <summary> The <see cref="ComputeMemory"/> will be accessible from the <see cref="ComputeKernel"/> for write operations only. </summary>
        WriteOnly = 1 << 1,
        /// <summary> The <see cref="ComputeMemory"/> will be accessible from the <see cref="ComputeKernel"/> for read operations only. </summary>
        ReadOnly = 1 << 2,
        /// <summary> </summary>
        UseHostPointer = 1 << 3,
        /// <summary> </summary>
        AllocateHostPointer = 1 << 4,
        /// <summary> </summary>
        CopyHostPointer = 1 << 5
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeImageChannelOrder : int
    {
        /// <summary> </summary>
        R = 0x10B0,
        /// <summary> </summary>
        A = 0x10B1,
        /// <summary> </summary>
        RG = 0x10B2,
        /// <summary> </summary>
        RA = 0x10B3,
        /// <summary> </summary>
        Rgb = 0x10B4,
        /// <summary> </summary>
        Rgba = 0x10B5,
        /// <summary> </summary>
        Bgra = 0x10B6,
        /// <summary> </summary>
        Argb = 0x10B7,
        /// <summary> </summary>
        Intensity = 0x10B8,
        /// <summary> </summary>
        Luminance = 0x10B9,
        /// <summary> </summary>
        Rx = 0x10BA,
        /// <summary> </summary>
        Rgx = 0x10BB,
        /// <summary> </summary>
        Rgbx = 0x10BC
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeImageChannelType : int
    {
        /// <summary> </summary>
        SNormInt8 = 0x10D0,
        /// <summary> </summary>
        SNormInt16 = 0x10D1,
        /// <summary> </summary>
        UNormInt8 = 0x10D2,
        /// <summary> </summary>
        UNormInt16 = 0x10D3,
        /// <summary> </summary>
        UNormShort565 = 0x10D4,
        /// <summary> </summary>
        UNormShort555 = 0x10D5,
        /// <summary> </summary>
        UNormInt101010 = 0x10D6,
        /// <summary> </summary>
        SignedInt8 = 0x10D7,
        /// <summary> </summary>
        SignedInt16 = 0x10D8,
        /// <summary> </summary>
        SignedInt32 = 0x10D9,
        /// <summary> </summary>
        UnsignedInt8 = 0x10DA,
        /// <summary> </summary>
        UnsignedInt16 = 0x10DB,
        /// <summary> </summary>
        UnsignedInt32 = 0x10DC,
        /// <summary> </summary>
        HalfFloat = 0x10DD,
        /// <summary> </summary>
        Float = 0x10DE,
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeMemoryType : int
    {
        /// <summary> </summary>
        Buffer = 0x10F0,
        /// <summary> </summary>
        Image2D = 0x10F1,
        /// <summary> </summary>
        Image3D = 0x10F2
    }

    /// <summary>
    /// The memory info query symbols.
    /// </summary>
    public enum ComputeMemoryInfo : int
    {
        /// <summary> </summary>
        Type = 0x1100,
        /// <summary> </summary>
        Flags = 0x1101,
        /// <summary> </summary>
        Size = 0x1102,
        /// <summary> </summary>
        HostPointer = 0x1103,
        /// <summary> </summary>
        MapppingCount = 0x1104,
        /// <summary> </summary>
        ReferenceCount = 0x1105,
        /// <summary> </summary>
        Context = 0x1106,
        /// <summary> </summary>
        AssociatedMemoryObject = 0x1107,
        /// <summary> </summary>
        Offset = 0x1108
    }

    /// <summary>
    /// The image info query symbols.
    /// </summary>
    public enum ComputeImageInfo : int
    {
        /// <summary> </summary>
        Format = 0x1110,
        /// <summary> </summary>
        ElementSize = 0x1111,
        /// <summary> </summary>
        RowPitch = 0x1112,
        /// <summary> </summary>
        SlicePitch = 0x1113,
        /// <summary> </summary>
        Width = 0x1114,
        /// <summary> </summary>
        Height = 0x1115,
        /// <summary> </summary>
        Depth = 0x1116
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeImageAddressing : int
    {
        /// <summary> </summary>
        None = 0x1130,
        /// <summary> </summary>
        ClampToEdge = 0x1131,
        /// <summary> </summary>
        Clamp = 0x1132,
        /// <summary> </summary>
        Repeat = 0x1133,
        /// <summary> </summary>
        MirroredRepeat = 0x1134
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeImageFiltering : int
    {
        /// <summary> </summary>
        Nearest = 0x1140,
        /// <summary> </summary>
        Linear = 0x1141
    }

    /// <summary>
    /// The sampler info query symbols.
    /// </summary>
    public enum ComputeSamplerInfo : int
    {
        /// <summary> </summary>
        ReferenceCount = 0x1150,
        /// <summary> </summary>
        Context = 0x1151,
        /// <summary> </summary>
        NormalizedCoords = 0x1152,
        /// <summary> </summary>
        Addressing = 0x1153,
        /// <summary> </summary>
        Filtering = 0x1154
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum ComputeMemoryMappingFlags : long
    {
        /// <summary> </summary>
        Read = 1 << 0,
        /// <summary> </summary>
        Write = 1 << 1
    }

    /// <summary>
    /// The program info query symbols.
    /// </summary>
    public enum ComputeProgramInfo : int
    {
        /// <summary> </summary>
        ReferenceCount = 0x1160,
        /// <summary> </summary>
        Context = 0x1161,
        /// <summary> </summary>
        DeviceCount = 0x1162,
        /// <summary> </summary>
        Devices = 0x1163,
        /// <summary> </summary>
        Source = 0x1164,
        /// <summary> </summary>
        BinarySizes = 0x1165,
        /// <summary> </summary>
        Binaries = 0x1166
    }

    /// <summary>
    /// The program build info query symbols.
    /// </summary>
    public enum ComputeProgramBuildInfo : int
    {
        /// <summary> </summary>
        Status = 0x1181,
        /// <summary> </summary>
        Options = 0x1182,
        /// <summary> </summary>
        BuildLog = 0x1183
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeProgramBuildStatus : int
    {
        /// <summary> </summary>
        Success = 0,
        /// <summary> </summary>
        None = -1,
        /// <summary> </summary>
        Error = -2,
        /// <summary> </summary>
        InProgress = -3
    }

    /// <summary>
    /// The kernel info query symbols.
    /// </summary>
    public enum ComputeKernelInfo : int
    {
        /// <summary> </summary>
        FunctionName = 0x1190,
        /// <summary> </summary>
        ArgumentCount = 0x1191,
        /// <summary> </summary>
        ReferenceCount = 0x1192,
        /// <summary> </summary>
        Context = 0x1193,
        /// <summary> </summary>
        Program = 0x1194
    }

    /// <summary>
    /// The kernel work-group info query symbols.
    /// </summary>
    public enum ComputeKernelWorkGroupInfo : int
    {
        /// <summary> </summary>
        WorkGroupSize = 0x11B0,
        /// <summary> </summary>
        CompileWorkGroupSize = 0x11B1,
        /// <summary> </summary>
        LocalMemorySize = 0x11B2,
        /// <summary> </summary>
        PreferredWorkGroupSizeMultiple = 0x11B3,
        /// <summary> </summary>
        PrivateMemorySize = 0x11B4
    }

    /// <summary>
    /// The event info query symbols.
    /// </summary>
    public enum ComputeEventInfo : int
    {
        /// <summary> </summary>
        CommandQueue = 0x11D0,
        /// <summary> </summary>
        CommandType = 0x11D1,
        /// <summary> </summary>
        ReferenceCount = 0x11D2,
        /// <summary> </summary>
        ExecutionStatus = 0x11D3,
        /// <summary> </summary>
        Context = 0x11D4
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeCommandType : int
    {
        /// <summary> </summary>
        NDRangeKernel = 0x11F0,
        /// <summary> </summary>
        Task = 0x11F1,
        /// <summary> </summary>
        NativeKernel = 0x11F2,
        /// <summary> </summary>
        ReadBuffer = 0x11F3,
        /// <summary> </summary>
        WriteBuffer = 0x11F4,
        /// <summary> </summary>
        CopyBuffer = 0x11F5,
        /// <summary> </summary>
        ReadImage = 0x11F6,
        /// <summary> </summary>
        WriteImage = 0x11F7,
        /// <summary> </summary>
        CopyImage = 0x11F8,
        /// <summary> </summary>
        CopyImageToBuffer = 0x11F9,
        /// <summary> </summary>
        CopyBufferToImage = 0x11FA,
        /// <summary> </summary>
        MapBuffer = 0x11FB,
        /// <summary> </summary>
        MapImage = 0x11FC,
        /// <summary> </summary>
        UnmapMemory = 0x11FD,
        /// <summary> </summary>
        Marker = 0x11FE,
        /// <summary> </summary>
        AcquireGLObjects = 0x11FF,
        /// <summary> </summary>
        ReleaseGLObjects = 0x1200,
        /// <summary> </summary>
        ReadBufferRectangle = 0x1201,
        /// <summary> </summary>
        WriteBufferRectangle = 0x1202,
        /// <summary> </summary>
        CopyBufferRectangle = 0x1203,
        /// <summary> </summary>
        User = 0x1204,
        /// <summary> </summary>
        CL_COMMAND_MIGRATE_MEM_OBJECT_EXT = 0x4040
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeCommandExecutionStatus : int
    {
        /// <summary> </summary>
        Complete = 0x0,
        /// <summary> </summary>
        Running = 0x1,
        /// <summary> </summary>
        Submitted = 0x2,
        /// <summary> </summary>
        Queued = 0x3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeBufferCreateType : int
    {
        /// <summary> </summary>
        Region = 0x1220
    }

    /// <summary>
    /// The command profiling info query symbols.
    /// </summary>
    public enum ComputeCommandProfilingInfo : int
    {
        /// <summary> </summary>
        Queued = 0x1280,
        /// <summary> </summary>
        Submitted = 0x1281,
        /// <summary> </summary>
        Started = 0x1282,
        /// <summary> </summary>
        Ended = 0x1283
    }

    /**************************************************************************************/
    // CL/GL Sharing API

    /// <summary>
    /// 
    /// </summary>
    public enum ComputeGLObjectType : int
    {
        /// <summary> </summary>
        Buffer = 0x2000,
        /// <summary> </summary>
        Texture2D = 0x2001,
        /// <summary> </summary>
        Texture3D = 0x2002,
        /// <summary> </summary>
        Renderbuffer = 0x2003
    }

    /// <summary>
    /// The shared CL/GL image/texture info query symbols.
    /// </summary>
    public enum ComputeGLTextureInfo : int
    {
        /// <summary> </summary>
        TextureTarget = 0x2004,
        /// <summary> </summary>
        MipMapLevel = 0x2005
    }

    /// <summary>
    /// The shared CL/GL context info query symbols.
    /// </summary>
    public enum ComputeGLContextInfo : int
    {
        /// <summary> </summary>
        CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR = 0x2006,
        /// <summary> </summary>
        CL_DEVICES_FOR_GL_CONTEXT_KHR = 0x2007
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum cl_device_partition_property_ext
    {
        /// <summary> </summary>
        CL_DEVICE_PARTITION_EQUALLY_EXT = 0x4050,
        /// <summary> </summary>
        CL_DEVICE_PARTITION_BY_COUNTS_EXT = 0x4051,
        /// <summary> </summary>
        CL_DEVICE_PARTITION_BY_NAMES_EXT = 0x4052,
        /// <summary> </summary>
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT = 0x4053,

        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_L1_CACHE_EXT = 0x1,
        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_L2_CACHE_EXT = 0x2,
        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_L3_CACHE_EXT = 0x3,
        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_L4_CACHE_EXT = 0x4,
        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_NUMA_EXT = 0x10,
        /// <summary> </summary>
        CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT = 0x100,

        /// <summary> </summary>
        CL_PROPERTIES_LIST_END_EXT = 0x0,
        /// <summary> </summary>
        CL_PARTITION_BY_COUNTS_LIST_END_EXT = 0x0,
        /// <summary> </summary>
        CL_PARTITION_BY_NAMES_LIST_END_EXT = -1
    }

    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum cl_mem_migration_flags_ext
    {
        /// <summary> </summary>
        None = 0,
        /// <summary> </summary>
        CL_MIGRATE_MEM_OBJECT_HOST_EXT = 0x1,
    }

    internal enum CLFunctionNames
    {
        Unknown,
        GetPlatformIDs,
        GetPlatformInfo,
        GetDeviceIDs,
        GetDeviceInfo,
        CreateContext,
        CreateContextFromType,
        RetainContext,
        ReleaseContext,
        GetContextInfo,
        CreateCommandQueue,
        RetainCommandQueue,
        ReleaseCommandQueue,
        GetCommandQueueInfo,
        SetCommandQueueProperty,
        CreateBuffer,
        CreateImage2D,
        CreateImage3D,
        RetainMemObject,
        ReleaseMemObject,
        GetSupportedImageFormats,
        GetMemObjectInfo,
        GetImageInfo,
        CreateSampler,
        RetainSampler,
        ReleaseSampler,
        GetSamplerInfo,
        CreateProgramWithSource,
        CreateProgramWithBinary,
        RetainProgram,
        ReleaseProgram,
        BuildProgram,
        UnloadCompiler,
        GetProgramInfo,
        GetProgramBuildInfo,
        CreateKernel,
        CreateKernelsInProgram,
        RetainKernel,
        ReleaseKernel,
        SetKernelArg,
        GetKernelInfo,
        GetKernelWorkGroupInfo,
        WaitForEvents,
        GetEventInfo,
        RetainEvent,
        ReleaseEvent,
        GetEventProfilingInfo,
        Flush,
        Finish,
        EnqueueReadBuffer,
        EnqueueWriteBuffer,
        EnqueueCopyBuffer,
        EnqueueReadImage,
        EnqueueWriteImage,
        EnqueueCopyImage,
        EnqueueCopyImageToBuffer,
        EnqueueCopyBufferToImage,
        EnqueueMapBuffer,
        EnqueueMapImage,
        EnqueueUnmapMemObject,
        EnqueueNDRangeKernel,
        EnqueueTask,
        EnqueueNativeKernel,
        EnqueueMarker,
        EnqueueWaitForEvents,
        EnqueueBarrier,
        GetExtensionFunctionAddress,
        CreateFromGLBuffer,
        CreateFromGLTexture2D,
        CreateFromGLTexture3D,
        CreateFromGLRenderbuffer,
        GetGLObjectInfo,
        GetGLTextureInfo,
        EnqueueAcquireGLObjects,
        EnqueueReleaseGLObjects,
    }
}