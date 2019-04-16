using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading;
using Cloo.Bindings;

namespace Cloo
{
    public class ComputeContext : ComputeResource
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputePlatform platform;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContextPropertyList properties;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeContextNotifier callback;

        public CLContextHandle Handle
        {
            get;
            protected set;
        }

        public ReadOnlyCollection<ComputeDevice> Devices { get { return devices; } }

        public ComputeContext(ICollection<ComputeDevice> devices, ComputeContextPropertyList properties, ComputeContextNotifier notify, IntPtr notifyDataPtr)
        {
            int handleCount;
            CLDeviceHandle[] deviceHandles = ComputeTools.ExtractHandles(devices, out handleCount);
            IntPtr[] propertyArray = (properties != null) ? properties.ToIntPtrArray() : null;
            callback = notify;

            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateContext(propertyArray, handleCount, deviceHandles, notify, notifyDataPtr, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            this.properties = properties;
            ComputeContextProperty platformProperty = properties.GetByName(ComputeContextPropertyName.Platform);
            this.platform = ComputePlatform.GetByHandle(platformProperty.Value);
            this.devices = GetDevices();

#if DEBUG
            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
        }

        protected override void Dispose(bool manual)
        {
            if (manual)
            {
                //free managed resources
            }

            // free native resources
            if (Handle.IsValid)
            {
#if DEBUG
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
#endif
                CL10.ReleaseContext(Handle);
                Handle.Invalidate();
            }
        }

        private ReadOnlyCollection<ComputeDevice> GetDevices()
        {
            List<CLDeviceHandle> deviceHandles = new List<CLDeviceHandle>(GetArrayInfo<CLContextHandle, ComputeContextInfo, CLDeviceHandle>(Handle, ComputeContextInfo.Devices, CL10.GetContextInfo));
            List<ComputeDevice> devices = new List<ComputeDevice>();
            foreach (ComputePlatform platform in ComputePlatform.Platforms)
            {
                foreach (ComputeDevice device in platform.Devices)
                    if (deviceHandles.Contains(device.Handle))
                        devices.Add(device);
            }
            return new ReadOnlyCollection<ComputeDevice>(devices);
        }
    }
}