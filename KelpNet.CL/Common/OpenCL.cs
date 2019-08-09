using System;
using System.IO;
using System.Linq;
using KelpNet.CL.Common;

namespace KelpNet.CL
{
    //GPU関連の処理を担うマネージャー
    public class OpenCL
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
        public static ComputeCommandQueue CommandQueue;
        public static int PlatformId;
        public static int DeviceIndex;
        public static bool Enable;
        public static string DeviceType;
        public static string DeviceName;
        public static string InfoString = ".Net Framework";

        public static string GetKernelSource(byte[] binary)
        {
            string result;

            using (StreamReader reader = new StreamReader(new MemoryStream(binary)))
            {
                result = reader.ReadToEnd();
            }

            return result;
        }

        public static void Initialize(int deviceIndex = 0)
        {
            //最初に見つかったプラットフォームを取得する
            for (int i = 0; i < ComputePlatform.Platforms.Count; i++)
            {
                if (ComputePlatform.Platforms[i].Devices.Count > 0)
                {
                    Initialize(i, deviceIndex);
                    return;
                }
            }
        }

        public static void Initialize(ComputeDeviceTypes selectedComputeDeviceTypes, int deviceIndex = 0)
        {
            //最初に見つかったプラットフォームを採用する
            for (int i = 0; i < ComputePlatform.Platforms.Count; i++)
            {
                var checklist = ComputePlatform.Platforms[i].Devices.Where(d => (long)d.Type == (long)selectedComputeDeviceTypes).ToArray();

                if (checklist.Length > 0)
                {
                    Initialize(i, deviceIndex);
                    return;
                }
            }
        }

        public static void Initialize(int platformId, int deviceIndex)
        {
            if (ComputePlatform.Platforms[platformId].Devices.Count > 0)
            {
                PlatformId = platformId;
                DeviceIndex = deviceIndex;

                DeviceType = ComputePlatform.Platforms[platformId].Devices[DeviceIndex].Type.ToString();
                DeviceName = ComputePlatform.Platforms[platformId].Devices[DeviceIndex].Name;

                Context = new ComputeContext(
                    ComputePlatform.Platforms[platformId].Devices,
                    new ComputeContextPropertyList(ComputePlatform.Platforms[platformId]),
                    null,
                    IntPtr.Zero
                    );

                CommandQueue = new ComputeCommandQueue(
                    Context,
                    ComputePlatform.Platforms[platformId].Devices[DeviceIndex],
                    ComputeCommandQueueFlags.None
                    );

                Enable = true;
                InfoString = GetInfo();
            }
        }

        public static ComputeProgram CreateProgram(string source)
        {
            string realType = Real.Size == sizeof(double) ? "double" : "float";

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
                program.Build(ComputePlatform.Platforms[PlatformId].Devices, string.Format("-D REAL={0} -Werror", realType), null, IntPtr.Zero);
            }
            catch (Exception e)
            {
                throw new Exception(program.GetBuildLog(ComputePlatform.Platforms[PlatformId].Devices[DeviceIndex]), e);
            }

            return program;
        }

        private static string GetInfo()
        {
            if (OpenCL.Enable)
            {
                return "OpenCL[" + OpenCL.DeviceType + "]: " + OpenCL.DeviceName;
            }
            else
            {
                return ".Net Framework";
            }
        }
    }
}
