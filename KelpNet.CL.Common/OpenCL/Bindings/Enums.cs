using System;

namespace KelpNet.CL.Common
{
    public enum ComputePlatformInfo : int
    {
        Profile = 0x0900,
        Version = 0x0901,
        Name = 0x0902,
        Vendor = 0x0903,
        Extensions = 0x0904,
        CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920,
    }

    [Flags]
    public enum ComputeDeviceTypes : long
    {
        Default = 1 << 0,
        Cpu = 1 << 1,
        Gpu = 1 << 2,
        Accelerator = 1 << 3,
        All = 0xFFFFFFFF
    }

    public enum ComputeDeviceInfo : int
    {
        Type = 0x1000,
        VendorId = 0x1001,
        MaxComputeUnits = 0x1002,
        MaxWorkItemDimensions = 0x1003,
        MaxWorkGroupSize = 0x1004,
        MaxWorkItemSizes = 0x1005,
        PreferredVectorWidthChar = 0x1006,
        PreferredVectorWidthShort = 0x1007,
        PreferredVectorWidthInt = 0x1008,
        PreferredVectorWidthLong = 0x1009,
        PreferredVectorWidthFloat = 0x100A,
        PreferredVectorWidthDouble = 0x100B,
        MaxClockFrequency = 0x100C,
        AddressBits = 0x100D,
        MaxReadImageArguments = 0x100E,
        MaxWriteImageArguments = 0x100F,
        MaxMemoryAllocationSize = 0x1010,
        Image2DMaxWidth = 0x1011,
        Image2DMaxHeight = 0x1012,
        Image3DMaxWidth = 0x1013,
        Image3DMaxHeight = 0x1014,
        Image3DMaxDepth = 0x1015,
        ImageSupport = 0x1016,
        MaxParameterSize = 0x1017,
        MaxSamplers = 0x1018,
        MemoryBaseAddressAlignment = 0x1019,
        MinDataTypeAlignmentSize = 0x101A,
        SingleFPConfig = 0x101B,
        GlobalMemoryCacheType = 0x101C,
        GlobalMemoryCachelineSize = 0x101D,
        GlobalMemoryCacheSize = 0x101E,
        GlobalMemorySize = 0x101F,
        MaxConstantBufferSize = 0x1020,
        MaxConstantArguments = 0x1021,
        LocalMemoryType = 0x1022,
        LocalMemorySize = 0x1023,
        ErrorCorrectionSupport = 0x1024,
        ProfilingTimerResolution = 0x1025,
        EndianLittle = 0x1026,
        Available = 0x1027,
        CompilerAvailable = 0x1028,
        ExecutionCapabilities = 0x1029,
        CommandQueueProperties = 0x102A,
        Name = 0x102B,
        Vendor = 0x102C,
        DriverVersion = 0x102D,
        Profile = 0x102E,
        Version = 0x102F,
        Extensions = 0x1030,
        Platform = 0x1031,
        CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032,
        CL_DEVICE_HALF_FP_CONFIG = 0x1033,
        PreferredVectorWidthHalf = 0x1034,
        HostUnifiedMemory = 0x1035,
        NativeVectorWidthChar = 0x1036,
        NativeVectorWidthShort = 0x1037,
        NativeVectorWidthInt = 0x1038,
        NativeVectorWidthLong = 0x1039,
        NativeVectorWidthFloat = 0x103A,
        NativeVectorWidthDouble = 0x103B,
        NativeVectorWidthHalf = 0x103C,
        OpenCLCVersion = 0x103D,
        CL_DEVICE_PARENT_DEVICE_EXT = 0x4054,
        CL_DEVICE_PARITION_TYPES_EXT = 0x4055,
        CL_DEVICE_AFFINITY_DOMAINS_EXT = 0x4056,
        CL_DEVICE_REFERENCE_COUNT_EXT = 0x4057,
        CL_DEVICE_PARTITION_STYLE_EXT = 0x4058
    }


    [Flags]
    public enum ComputeCommandQueueFlags : long
    {
        None = 0,
        OutOfOrderExecution = 1 << 0,
        Profiling = 1 << 1
    }

    public enum ComputeContextInfo : int
    {
        ReferenceCount = 0x1080,
        Devices = 0x1081,
        Properties = 0x1082,
        NumDevices = 0x1083,
        Platform = 0x1084,
    }

    public enum ComputeContextPropertyName : int
    {
        Platform = ComputeContextInfo.Platform,
        CL_GL_CONTEXT_KHR = 0x2008,
        CL_EGL_DISPLAY_KHR = 0x2009,
        CL_GLX_DISPLAY_KHR = 0x200A,
        CL_WGL_HDC_KHR = 0x200B,
        CL_CGL_SHAREGROUP_KHR = 0x200C,
    }

    [Flags]
    public enum ComputeMemoryFlags : long
    {
        None = 0,
        ReadWrite = 1 << 0,
        WriteOnly = 1 << 1,
        ReadOnly = 1 << 2,
        UseHostPointer = 1 << 3,
        AllocateHostPointer = 1 << 4,
        CopyHostPointer = 1 << 5
    }

    public enum ComputeMemoryInfo : int
    {
        Type = 0x1100,
        Flags = 0x1101,
        Size = 0x1102,
        HostPointer = 0x1103,
        MapppingCount = 0x1104,
        ReferenceCount = 0x1105,
        Context = 0x1106,
        AssociatedMemoryObject = 0x1107,
        Offset = 0x1108
    }

    public enum ComputeProgramBuildInfo : int
    {
        Status = 0x1181,
        Options = 0x1182,
        BuildLog = 0x1183
    }

    public enum ComputeEventInfo : int
    {
        CommandQueue = 0x11D0,
        CommandType = 0x11D1,
        ReferenceCount = 0x11D2,
        ExecutionStatus = 0x11D3,
        Context = 0x11D4
    }

    public enum ComputeCommandType : int
    {
        NDRangeKernel = 0x11F0,
        Task = 0x11F1,
        NativeKernel = 0x11F2,
        ReadBuffer = 0x11F3,
        WriteBuffer = 0x11F4,
        CopyBuffer = 0x11F5,
        ReadImage = 0x11F6,
        WriteImage = 0x11F7,
        CopyImage = 0x11F8,
        CopyImageToBuffer = 0x11F9,
        CopyBufferToImage = 0x11FA,
        MapBuffer = 0x11FB,
        MapImage = 0x11FC,
        UnmapMemory = 0x11FD,
        Marker = 0x11FE,
        AcquireGLObjects = 0x11FF,
        ReleaseGLObjects = 0x1200,
        ReadBufferRectangle = 0x1201,
        WriteBufferRectangle = 0x1202,
        CopyBufferRectangle = 0x1203,
        User = 0x1204,
        CL_COMMAND_MIGRATE_MEM_OBJECT_EXT = 0x4040
    }

    public enum ComputeCommandExecutionStatus : int
    {
        Complete = 0x0,
        Running = 0x1,
        Submitted = 0x2,
        Queued = 0x3
    }
}