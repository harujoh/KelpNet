using System;
using System.Linq;
using System.Runtime.InteropServices;
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
        Default = 1 << 0,
        Cpu = 1 << 1,
        Gpu = 1 << 2,
        Accelerator = 1 << 3,
        All = 0xFFFFFFFF
    }

    //GPU関連の処理を担うマネージャー
    public class Weaver
    {
        public const string USE_DOUBLE_HEADER_STRING =
@"
#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#endif
";

        public const string REAL_HEADER_STRING =
@"
//! REAL is provided by compiler option
typedef REAL Real;
";

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

        public static ComputeKernel CreateKernel(string source, string kenrlName)
        {
            return CreateProgram(source).CreateKernel(kenrlName);
        }

        public static ComputeProgram CreateProgram(string source)
        {
            string realType = Marshal.SizeOf(typeof(Real)) == Marshal.SizeOf(typeof(double)) ? "double" : "float";

            //浮動小数点の精度設定用
            source = REAL_HEADER_STRING + source;

            //倍精度時に追加
            if (realType == "double")
            {
                source = USE_DOUBLE_HEADER_STRING + source;
            }

            ComputeProgram program = new ComputeProgram(Context, source);

            try
            {
                program.Build(Devices, string.Format("-D REAL={0} -Werror", realType), null, IntPtr.Zero);
            }
            catch
            {
                MessageBox.Show(program.GetBuildLog(Devices[DeviceIndex]));
            }

            return program;
        }
    }
}
