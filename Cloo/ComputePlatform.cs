using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputePlatform : ComputeObject
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<string> extensions;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private static ReadOnlyCollection<ComputePlatform> platforms;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string profile;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string vendor;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string version;


        public CLPlatformHandle Handle
        {
            get;
            protected set;
        }

        public ReadOnlyCollection<ComputeDevice> Devices { get { return devices; } }

        public string Name { get { return name; } }

        public static ReadOnlyCollection<ComputePlatform> Platforms { get { return platforms; } }

        public string Version { get { return version; } }

        static ComputePlatform()
        {
            lock (typeof(ComputePlatform))
            {
                try
                {
                    if (platforms != null)
                        return;

                    CLPlatformHandle[] handles;
                    int handlesLength;
                    ComputeErrorCode error = CL10.GetPlatformIDs(0, null, out handlesLength);
                    ComputeException.ThrowOnError(error);
                    handles = new CLPlatformHandle[handlesLength];

                    error = CL10.GetPlatformIDs(handlesLength, handles, out handlesLength);
                    ComputeException.ThrowOnError(error);

                    List<ComputePlatform> platformList = new List<ComputePlatform>(handlesLength);
                    foreach (CLPlatformHandle handle in handles)
                        platformList.Add(new ComputePlatform(handle));

                    platforms = platformList.AsReadOnly();
                }
                catch (DllNotFoundException)
                {
                    platforms = new List<ComputePlatform>().AsReadOnly();
                }
            }
        }

        private ComputePlatform(CLPlatformHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            string extensionString = GetStringInfo<CLPlatformHandle, ComputePlatformInfo>(Handle, ComputePlatformInfo.Extensions, CL10.GetPlatformInfo);
            extensions = new ReadOnlyCollection<string>(extensionString.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

            name = GetStringInfo<CLPlatformHandle, ComputePlatformInfo>(Handle, ComputePlatformInfo.Name, CL10.GetPlatformInfo);
            profile = GetStringInfo<CLPlatformHandle, ComputePlatformInfo>(Handle, ComputePlatformInfo.Profile, CL10.GetPlatformInfo);
            vendor = GetStringInfo<CLPlatformHandle, ComputePlatformInfo>(Handle, ComputePlatformInfo.Vendor, CL10.GetPlatformInfo);
            version = GetStringInfo<CLPlatformHandle, ComputePlatformInfo>(Handle, ComputePlatformInfo.Version, CL10.GetPlatformInfo);
            QueryDevices();
        }

        public static ComputePlatform GetByHandle(IntPtr handle)
        {
            foreach (ComputePlatform platform in Platforms)
                if (platform.Handle.Value == handle)
                    return platform;

            return null;
        }

        public ReadOnlyCollection<ComputeDevice> QueryDevices()
        {
            int handlesLength = 0;
            ComputeErrorCode error = CL10.GetDeviceIDs(Handle, ComputeDeviceTypes.All, 0, null, out handlesLength);
            ComputeException.ThrowOnError(error);

            CLDeviceHandle[] handles = new CLDeviceHandle[handlesLength];
            error = CL10.GetDeviceIDs(Handle, ComputeDeviceTypes.All, handlesLength, handles, out handlesLength);
            ComputeException.ThrowOnError(error);

            ComputeDevice[] devices = new ComputeDevice[handlesLength];
            for (int i = 0; i < handlesLength; i++)
                devices[i] = new ComputeDevice(this, handles[i]);

            this.devices = new ReadOnlyCollection<ComputeDevice>(devices);

            return this.devices;
        }
    }
}