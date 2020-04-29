using System;
using System.Collections.Generic;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
#else
using KelpNet.CL.Properties;
using Real = System.Single;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    public static class CompressibleActivation
    {
        public static bool SetParallel<T>(this ICompressibleActivation<T> compressibleActivation, bool enable, KeyValuePair<string, string>[] activationParameters = null) where T : unmanaged, IComparable<T>
        {
            compressibleActivation.IsParallel = enable & OpenCL.Enable;

            if (compressibleActivation.IsParallel)
            {
                string kernelNameBase = compressibleActivation.FunctionName.Replace(" ", "");
                compressibleActivation.ActivateKernelString = OpenCL.GetKernelSource(Resources.Activation).Replace("/*kernelNameBase*/", kernelNameBase);
                compressibleActivation.ForwardKernelName = kernelNameBase + "Forward";
                compressibleActivation.BackwardKernelName = kernelNameBase + "Backward";

                string kernelSource = compressibleActivation.KernelSource;

                if (activationParameters != null)
                {
                    foreach (var parameter in activationParameters)
                    {
                        kernelSource = kernelSource.Replace(parameter.Key, parameter.Value);
                    }
                }

                kernelSource += compressibleActivation.ActivateKernelString;

                ComputeProgram program = OpenCL.CreateProgram<T>(kernelSource);
                compressibleActivation.ForwardKernel = program.CreateKernel(compressibleActivation.ForwardKernelName);
                compressibleActivation.BackwardKernel = program.CreateKernel(compressibleActivation.BackwardKernelName);
            }

            return compressibleActivation.IsParallel;
        }
    }
#endif

#if DOUBLE
    public static class CompressibleActivationD
#else
    public static class CompressibleActivationF
#endif
    {
        public static NdArray<Real> NeedPreviousForwardGpu(this ICompressibleActivation<Real> compressibleActivation, NdArray<Real> x)
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

        public static void NeedPreviousBackwardGpu(this ICompressibleActivation<Real> compressibleActivation, NdArray<Real> y, NdArray<Real> x)
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
