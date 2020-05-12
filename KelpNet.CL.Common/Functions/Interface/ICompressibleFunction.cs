using System;
using KelpNet.CL.Common;

namespace KelpNet.CL
{
    public interface ICompressibleFunction<T> : CPU.ICompressibleFunction<T>, IParallelizable where T : unmanaged, IComparable<T>
    {
        ComputeKernel ForwardKernel { get; set; }
        ComputeKernel BackwardgWKernel { get; set; }
        ComputeKernel BackwardgXKernel { get; set; }

        string ForwardKernelName { get; set; }
        string BackwardgWKernelName { get; set; }
        string BackwardgXKernelName { get; set; }

        new ICompressibleActivation<T> Activation { get; set; }
    }

    public static class CompressibleFunction
    {
        public static bool SetParallel<T>(this ICompressibleFunction<T> compressibleFunction, bool enable) where T : unmanaged, IComparable<T>
        {
            compressibleFunction.IsParallel = enable & OpenCL.Enable;

            if (compressibleFunction.IsParallel)
            {
                string kernelNameBase = compressibleFunction.FunctionName.Replace(" ", "");
                compressibleFunction.ForwardKernelName = kernelNameBase + "Forward";
                compressibleFunction.BackwardgWKernelName = kernelNameBase + "gWBackward";
                compressibleFunction.BackwardgXKernelName = kernelNameBase + "gXBackward";

                string kernelSource = compressibleFunction.KernelSource;

                if (compressibleFunction.Activation is ICompressibleActivation<T> activator)
                {
                    //アクティベーションを活性化
                    kernelSource = activator.KernelSource + kernelSource.Replace("/*ForwardActivate*/", "ForwardActivate");
                }

                ComputeProgram program = OpenCL.CreateProgram<T>(kernelSource);
                compressibleFunction.ForwardKernel = program.CreateKernel(compressibleFunction.ForwardKernelName);
                compressibleFunction.BackwardgWKernel = program.CreateKernel(compressibleFunction.BackwardgWKernelName);
                compressibleFunction.BackwardgXKernel = program.CreateKernel(compressibleFunction.BackwardgXKernelName);
            }

            return compressibleFunction.IsParallel;
        }
    }
}
