using System;
using System.Linq;
using System.Windows.Forms;
using Cloo;

namespace KelpNet.Common
{
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

    //GPU関連の処理を担うマネージャー
    public class Weaver
    {
        public static ComputeContext Context;
        private static ComputeDevice[] Devices;
        public static ComputeCommandQueue CommandQueue;
        public static int DeviceIndex;
        public static bool Enable;
        public static ComputePlatform Platform;

        public static void Initialize(ComputeDeviceTypes selectedComputeDeviceTypes, int platformId = 0, int deviceIndex = 0)
        {
            Platform = ComputePlatform.Platforms[platformId];

            Devices = Platform
                .Devices
                .Where(d => (long)d.Type == (long)selectedComputeDeviceTypes)
                .ToArray();

            Context = new ComputeContext(
                Devices,
                new ComputeContextPropertyList(Platform),
                null,
                IntPtr.Zero
                );

            CommandQueue = new ComputeCommandQueue(
                Context,
                Devices[DeviceIndex],
                ComputeCommandQueueFlags.None
                );

            DeviceIndex = deviceIndex;

            Enable = true;
        }

        public static ComputeKernel CreateKernel(string source, string kernelName)
        {
            ComputeProgram program = new ComputeProgram(Context, source);

            try
            {
                program.Build(Devices, null, null, IntPtr.Zero);
            }
            catch
            {
                MessageBox.Show(program.GetBuildLog(Devices[DeviceIndex]));
            }

            return program.CreateKernel(kernelName);
        }
    }
}
