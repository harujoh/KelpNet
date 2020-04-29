using System;
using System.Diagnostics;

namespace KelpNet.CL.Common
{
    public class ComputeDevice : ComputeObject
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string name;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputePlatform platform;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceTypes type;

        public string Name { get { return name; } }

        public ComputePlatform Platform { get { return platform; } }

        public ComputeDeviceTypes Type { get { return type; } }

        internal ComputeDevice(ComputePlatform platform, IntPtr handle)
        {
            this.handle = handle;

            name = GetStringInfo(ComputeDeviceInfo.Name);
            this.platform = platform;
            type = (ComputeDeviceTypes)GetInfo<long>(ComputeDeviceInfo.Type);
        }

        private NativeType GetInfo<NativeType>(ComputeDeviceInfo paramName) where NativeType : struct
        {
            return GetInfo<IntPtr, ComputeDeviceInfo, NativeType>(handle, paramName, CL10.GetDeviceInfo);
        }

        private string GetStringInfo(ComputeDeviceInfo paramName)
        {
            return GetStringInfo(handle, paramName, CL10.GetDeviceInfo);
        }
    }
}