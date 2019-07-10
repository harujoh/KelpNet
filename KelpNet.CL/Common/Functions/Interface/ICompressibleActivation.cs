using System.Collections.Generic;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    public interface ICompressibleActivation : KelpNet.ICompressibleActivation, IParallelizable
    {
        ComputeKernel ForwardKernel { get; set; }
        ComputeKernel BackwardKernel { get; set; }

        //GPU向けのActivate関数の文字列
        string ActivateFunctionString { get; set; } //外部の関数に組み込まれる
        string ActivateKernelString { get; set; } //単品で呼ぶ用

        KeyValuePair<string, string>[] ActivationParameters { get; set; }

        string ForwardKernelName { get; set; }
        string BackwardKernelName { get; set; }
    }

    public static class ICompressibleActivationFunction
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public static void Initialize(this ICompressibleActivation compressibleActivation, string functionName, string activateFunctionString, KeyValuePair<string, string>[] parameters, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false)
        {
            string kernelNameBase = functionName.Replace(" ", "");

            compressibleActivation.ActivateFunctionString = activateFunctionString;

            compressibleActivation.ActivateKernelString = OpenCL.GetKernelSource(Resources.Activation).Replace("/*kernelNameBase*/", kernelNameBase);

            if (parameters == null)
            {
                compressibleActivation.ActivationParameters = new KeyValuePair<string, string>[] { };
            }
            else
            {
                compressibleActivation.ActivationParameters = parameters;
            }

            compressibleActivation.ForwardKernelName = kernelNameBase + "Forward";
            compressibleActivation.BackwardKernelName = kernelNameBase + "Backward";

            compressibleActivation.SetParallel(gpuEnable);
        }

        public static bool SetParallel(this ICompressibleActivation compressibleActivation, bool enable)
        {
            compressibleActivation.IsParallel = enable & OpenCL.Enable;

            if (compressibleActivation.IsParallel)
            {
                compressibleActivation.InitParallel();

                compressibleActivation.SingleInputForward = compressibleActivation.NeedPreviousForwardGpu;
                compressibleActivation.SingleOutputBackward = compressibleActivation.NeedPreviousBackwardGpu;
            }
            else
            {
                compressibleActivation.SingleInputForward = compressibleActivation.NeedPreviousForwardCpu;
                compressibleActivation.SingleOutputBackward = compressibleActivation.NeedPreviousBackwardCpu;
            }

            return compressibleActivation.IsParallel;
        }

        public static void InitParallel(this ICompressibleActivation compressibleActivation)
        {
            if (compressibleActivation.IsParallel)
            {
                string kernelSource = compressibleActivation.ActivateFunctionString;

                foreach (var parameter in compressibleActivation.ActivationParameters)
                {
                    kernelSource = compressibleActivation.ActivateFunctionString.Replace(parameter.Key, parameter.Value);
                }

                kernelSource += compressibleActivation.ActivateKernelString;

                ComputeProgram program = OpenCL.CreateProgram(kernelSource);
                compressibleActivation.ForwardKernel = program.CreateKernel(compressibleActivation.ForwardKernelName);
                compressibleActivation.BackwardKernel = program.CreateKernel(compressibleActivation.BackwardKernelName);
            }
        }

        private static NdArray NeedPreviousForwardGpu(this ICompressibleActivation compressibleActivation, NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, y.Length))
            {
                compressibleActivation.ForwardKernel.SetMemoryArgument(0, gpuX);
                compressibleActivation.ForwardKernel.SetMemoryArgument(1, gpuY);

                OpenCL.CommandQueue.Execute
                    (
                        compressibleActivation.ForwardKernel,
                        null,
                        new long[] { x.Data.Length },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, compressibleActivation);
        }

        private static void NeedPreviousBackwardGpu(this ICompressibleActivation compressibleActivation, NdArray y, NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, y.Grad))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, y.Data))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
            {
                compressibleActivation.BackwardKernel.SetMemoryArgument(0, gpugY);
                compressibleActivation.BackwardKernel.SetMemoryArgument(1, gpuY);
                compressibleActivation.BackwardKernel.SetMemoryArgument(2, gpugX);

                OpenCL.CommandQueue.Execute
                    (
                        compressibleActivation.BackwardKernel,
                        null,
                        new long[] { y.Grad.Length },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }

    }
}
