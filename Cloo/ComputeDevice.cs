using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeDevice : ComputeObject
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long addressBits;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool available;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool compilerAvailable;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string driverVersion;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool endianLittle;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool errorCorrectionSupport;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceExecutionCapabilities executionCapabilities;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<string> extensions;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCachelineSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCacheSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceMemoryCacheType globalMemoryCacheType;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemorySize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool imageSupport;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxHeight;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxWidth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxDepth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxHeight;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxWidth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long localMemorySize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceLocalMemoryType localMemoryType;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxClockFrequency;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxComputeUnits;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantArguments;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantBufferSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxMemAllocSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxParameterSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxReadImageArgs;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxSamplers;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkGroupSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkItemDimensions;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<long> maxWorkItemSizes;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWriteImageArgs;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long memBaseAddrAlign;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long minDataTypeAlignSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string name;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputePlatform platform;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthChar;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthFloat;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthInt;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthLong;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthShort;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string profile;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long profilingTimerResolution;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeCommandQueueFlags queueProperties;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceSingleCapabilities singleCapabilities;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceTypes type;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string vendor;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long vendorId;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string version;

        public CLDeviceHandle Handle
        {
            get;
            protected set;
        }

        public string Name { get { return name; } }

        public ComputePlatform Platform { get { return platform; } }

        public ComputeDeviceTypes Type { get { return type; } }

        internal ComputeDevice(ComputePlatform platform, CLDeviceHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            addressBits = GetInfo<uint>(ComputeDeviceInfo.AddressBits);
            available = GetBoolInfo(ComputeDeviceInfo.Available);
            compilerAvailable = GetBoolInfo(ComputeDeviceInfo.CompilerAvailable);
            driverVersion = GetStringInfo(ComputeDeviceInfo.DriverVersion);
            endianLittle = GetBoolInfo(ComputeDeviceInfo.EndianLittle);
            errorCorrectionSupport = GetBoolInfo(ComputeDeviceInfo.ErrorCorrectionSupport);
            executionCapabilities = (ComputeDeviceExecutionCapabilities)GetInfo<long>(ComputeDeviceInfo.ExecutionCapabilities);

            string extensionString = GetStringInfo(ComputeDeviceInfo.Extensions);
            extensions = new ReadOnlyCollection<string>(extensionString.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

            globalMemoryCachelineSize = GetInfo<uint>(ComputeDeviceInfo.GlobalMemoryCachelineSize);
            globalMemoryCacheSize = (long)GetInfo<ulong>(ComputeDeviceInfo.GlobalMemoryCacheSize);
            globalMemoryCacheType = (ComputeDeviceMemoryCacheType)GetInfo<long>(ComputeDeviceInfo.GlobalMemoryCacheType);
            globalMemorySize = (long)GetInfo<ulong>(ComputeDeviceInfo.GlobalMemorySize);
            image2DMaxHeight = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image2DMaxHeight);
            image2DMaxWidth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image2DMaxWidth);
            image3DMaxDepth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxDepth);
            image3DMaxHeight = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxHeight);
            image3DMaxWidth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxWidth);
            imageSupport = GetBoolInfo(ComputeDeviceInfo.ImageSupport);
            localMemorySize = (long)GetInfo<ulong>(ComputeDeviceInfo.LocalMemorySize);
            localMemoryType = (ComputeDeviceLocalMemoryType)GetInfo<long>(ComputeDeviceInfo.LocalMemoryType);
            maxClockFrequency = GetInfo<uint>(ComputeDeviceInfo.MaxClockFrequency);
            maxComputeUnits = GetInfo<uint>(ComputeDeviceInfo.MaxComputeUnits);
            maxConstantArguments = GetInfo<uint>(ComputeDeviceInfo.MaxConstantArguments);
            maxConstantBufferSize = (long)GetInfo<ulong>(ComputeDeviceInfo.MaxConstantBufferSize);
            maxMemAllocSize = (long)GetInfo<ulong>(ComputeDeviceInfo.MaxMemoryAllocationSize);
            maxParameterSize = (long)GetInfo<IntPtr>(ComputeDeviceInfo.MaxParameterSize);
            maxReadImageArgs = GetInfo<uint>(ComputeDeviceInfo.MaxReadImageArguments);
            maxSamplers = GetInfo<uint>(ComputeDeviceInfo.MaxSamplers);
            maxWorkGroupSize = (long)GetInfo<IntPtr>(ComputeDeviceInfo.MaxWorkGroupSize);
            maxWorkItemDimensions = GetInfo<uint>(ComputeDeviceInfo.MaxWorkItemDimensions);
            maxWorkItemSizes = new ReadOnlyCollection<long>(ComputeTools.ConvertArray(GetArrayInfo<CLDeviceHandle, ComputeDeviceInfo, IntPtr>(Handle, ComputeDeviceInfo.MaxWorkItemSizes, CL10.GetDeviceInfo)));
            maxWriteImageArgs = GetInfo<uint>(ComputeDeviceInfo.MaxWriteImageArguments);
            memBaseAddrAlign = GetInfo<uint>(ComputeDeviceInfo.MemoryBaseAddressAlignment);
            minDataTypeAlignSize = GetInfo<uint>(ComputeDeviceInfo.MinDataTypeAlignmentSize);
            name = GetStringInfo(ComputeDeviceInfo.Name);
            this.platform = platform;
            preferredVectorWidthChar = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthChar);
            preferredVectorWidthFloat = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthFloat);
            preferredVectorWidthInt = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthInt);
            preferredVectorWidthLong = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthLong);
            preferredVectorWidthShort = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthShort);
            profile = GetStringInfo(ComputeDeviceInfo.Profile);
            profilingTimerResolution = (long)GetInfo<IntPtr>(ComputeDeviceInfo.ProfilingTimerResolution);
            queueProperties = (ComputeCommandQueueFlags)GetInfo<long>(ComputeDeviceInfo.CommandQueueProperties);
            singleCapabilities = (ComputeDeviceSingleCapabilities)GetInfo<long>(ComputeDeviceInfo.SingleFPConfig);
            type = (ComputeDeviceTypes)GetInfo<long>(ComputeDeviceInfo.Type);
            vendor = GetStringInfo(ComputeDeviceInfo.Vendor);
            vendorId = GetInfo<uint>(ComputeDeviceInfo.VendorId);
            version = GetStringInfo(ComputeDeviceInfo.Version);
        }

        private bool GetBoolInfo(ComputeDeviceInfo paramName)
        {
            return GetBoolInfo<CLDeviceHandle, ComputeDeviceInfo>(Handle, paramName, CL10.GetDeviceInfo);
        }

        private NativeType GetInfo<NativeType>(ComputeDeviceInfo paramName) where NativeType : struct
        {
            return GetInfo<CLDeviceHandle, ComputeDeviceInfo, NativeType>(Handle, paramName, CL10.GetDeviceInfo);
        }

        private string GetStringInfo(ComputeDeviceInfo paramName)
        {
            return GetStringInfo<CLDeviceHandle, ComputeDeviceInfo>(Handle, paramName, CL10.GetDeviceInfo);
        }
    }
}