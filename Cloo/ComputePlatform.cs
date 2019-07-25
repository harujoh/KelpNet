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
        private readonly string name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private static ReadOnlyCollection<ComputePlatform> platforms;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string version;


        public CLPlatformHandle Handle { get; protected set; }

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

                    CLPlatformHandle[] handles;
                    int handlesLength;
                    ComputeErrorCode error = CL10.GetPlatformIDs(0, null, out handlesLength);
                    ComputeException.ThrowOnError(error);
                    handles = new CLPlatformHandle[handlesLength];

                    error = CL10.GetPlatformIDs(handlesLength, handles, out handlesLength);
                    ComputeException.ThrowOnError(error);

                    List<ComputePlatform> platformList = new List<ComputePlatform>(handlesLength);

                    foreach (CLPlatformHandle handle in handles)
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

        private ComputePlatform(CLPlatformHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            name = GetStringInfo(Handle, ComputePlatformInfo.Name, CL10.GetPlatformInfo);
            version = GetStringInfo(Handle, ComputePlatformInfo.Version, CL10.GetPlatformInfo);
            QueryDevices();
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
            {
                devices[i] = new ComputeDevice(this, handles[i]);
            }

            this.devices = new ReadOnlyCollection<ComputeDevice>(devices);

            return this.devices;
        }
    }
}