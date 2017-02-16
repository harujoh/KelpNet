using System;
using System.Linq;
using System.Windows.Forms;
using Cloo;

namespace KelpNet.Common
{
    //GPU関連の処理を担うマネージャー
    class Weaver
    {
        public static ComputeContext Context;
        private static ComputeDevice[] Devices;
        public  static ComputeCommandQueue CommandQueue;

        static Weaver()
        {
            ComputePlatform platform = ComputePlatform.Platforms[0];

            Devices = platform
                .Devices
                .Where(d => d.Type == ComputeDeviceTypes.Gpu)
                .ToArray();

            Context = new ComputeContext(
                Devices,
                new ComputeContextPropertyList(platform),
                null,
                IntPtr.Zero
            );

            CommandQueue = new ComputeCommandQueue(
                Context,
                Devices[0],
                ComputeCommandQueueFlags.None
            );
        }

        public static ComputeKernel CreateKernel(string source,string kernelName)
        {
            ComputeProgram program = new ComputeProgram(Context, source);

            try
            {
                program.Build(Devices, null, null, IntPtr.Zero);
            }
            catch
            {
                MessageBox.Show(program.GetBuildLog(Devices[0]));
            }

            return program.CreateKernel(kernelName);
        }
    }
}
