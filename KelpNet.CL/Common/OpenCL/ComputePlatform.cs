using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace KelpNet.CL.Common
{
    public class ComputePlatform : ComputeObject
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private static ReadOnlyCollection<ComputePlatform> platforms;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string version;

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
                    {
                        return;
                    }

                    IntPtr[] handles;
                    int handlesLength;
                    CL10.GetPlatformIDs(0, null, out handlesLength);
                    handles = new IntPtr[handlesLength];

                    CL10.GetPlatformIDs(handlesLength, handles, out handlesLength);

                    List<ComputePlatform> platformList = new List<ComputePlatform>(handlesLength);

                    foreach (IntPtr handle in handles)
                    {
                        platformList.Add(new ComputePlatform(handle));
                    }

                    platforms = platformList.AsReadOnly();
                }
                catch (DllNotFoundException)
                {
                    platforms = new List<ComputePlatform>().AsReadOnly();
                }
            }
        }

        private ComputePlatform(IntPtr handle)
        {
            this.handle = handle;

            name = GetStringInfo(handle, ComputePlatformInfo.Name, CL10.GetPlatformInfo);
            version = GetStringInfo(handle, ComputePlatformInfo.Version, CL10.GetPlatformInfo);
            QueryDevices();
        }

        public ReadOnlyCollection<ComputeDevice> QueryDevices()
        {
            int handlesLength = 0;
            CL10.GetDeviceIDs(handle, ComputeDeviceTypes.All, 0, null, out handlesLength);

            IntPtr[] handles = new IntPtr[handlesLength];
            CL10.GetDeviceIDs(handle, ComputeDeviceTypes.All, handlesLength, handles, out handlesLength);

            ComputeDevice[] devices = new ComputeDevice[handlesLength];

            for (int i = 0; i < handlesLength; i++)
            {
                devices[i] = new ComputeDevice(this, handles[i]);
            }

            this.devices = new ReadOnlyCollection<ComputeDevice>(devices);

            return this.devices;
        }
    }
}