using System.Diagnostics;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeDevice : ComputeObject
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string name;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputePlatform platform;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceTypes type;

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

            name = GetStringInfo(ComputeDeviceInfo.Name);
            this.platform = platform;
            type = (ComputeDeviceTypes)GetInfo<long>(ComputeDeviceInfo.Type);
        }

        private NativeType GetInfo<NativeType>(ComputeDeviceInfo paramName) where NativeType : struct
        {
            return GetInfo<CLDeviceHandle, ComputeDeviceInfo, NativeType>(Handle, paramName, CL10.GetDeviceInfo);
        }

        private string GetStringInfo(ComputeDeviceInfo paramName)
        {
            return GetStringInfo(Handle, paramName, CL10.GetDeviceInfo);
        }
    }
}