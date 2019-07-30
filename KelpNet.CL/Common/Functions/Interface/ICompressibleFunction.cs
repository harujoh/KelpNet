using Cloo;

namespace KelpNet.CL
{
    public interface ICompressibleFunction : KelpNet.ICompressibleFunction, IParallelizable
    {
        ComputeKernel ForwardKernel { get; set; }
        ComputeKernel BackwardgWKernel { get; set; }
        ComputeKernel BackwardgXKernel { get; set; }

        string ForwardKernelName { get; set; }
        string BackwardgWKernelName { get; set; }
        string BackwardgXKernelName { get; set; }

        string KernelString { get; set; }

        NdArray NeedPreviousForwardGpu(NdArray input);
        void NeedPreviousBackwardGpu(NdArray y, NdArray x);
    }

    public static class CompressibleFunction
    {
        public static void Initialize(this ICompressibleFunction compressibleFunction, string functionName, string kernelString, KelpNet.ICompressibleActivation activation = null, bool gpuEnable = false)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            compressibleFunction.ForwardKernelName = kernelNameBase + "Forward";
            compressibleFunction.BackwardgWKernelName = kernelNameBase + "gWBackward";
            compressibleFunction.BackwardgXKernelName = kernelNameBase + "gXBackward";

            compressibleFunction.Activation = activation;

            compressibleFunction.KernelString = kernelString;

            compressibleFunction.SetParallel(gpuEnable);
        }

        //後からActivationを追加する用
        public static void SetActivation(this ICompressibleFunction compressibleFunction, ICompressibleActivation activation)
        {
            compressibleFunction.Activation = activation;

            InitParallel(compressibleFunction);
        }

        public static bool SetParallel(this ICompressibleFunction compressibleFunction, bool enable)
        {
            compressibleFunction.IsParallel = enable & OpenCL.Enable;

            if (compressibleFunction.IsParallel)
            {
                compressibleFunction.InitParallel();

                compressibleFunction.SingleInputForward = compressibleFunction.NeedPreviousForwardGpu;
                compressibleFunction.SingleOutputBackward = compressibleFunction.NeedPreviousBackwardGpu;
            }
            else
            {
                compressibleFunction.SingleInputForward = compressibleFunction.NeedPreviousForwardCpu;
                compressibleFunction.SingleOutputBackward = compressibleFunction.NeedPreviousBackwardCpu;
            }

            return compressibleFunction.IsParallel;
        }

        public static void InitParallel(this ICompressibleFunction compressibleFunction)
        {
            if (compressibleFunction.IsParallel)
            {
                string kernelSource = compressibleFunction.KernelString;

                if (compressibleFunction.Activation is ICompressibleActivation activator)
                {
                    string activationSource = activator.ActivateFunctionString;

                    foreach (var activationParameter in activator.ActivationParameters)
                    {
                        activationSource = activationSource.Replace(activationParameter.Key, activationParameter.Value);
                    }

                    //アクティベーションを活性化
                    kernelSource = activationSource + kernelSource.Replace("/*ForwardActivate*/", "ForwardActivate");
                }

                ComputeProgram program = OpenCL.CreateProgram(kernelSource);
                compressibleFunction.ForwardKernel = program.CreateKernel(compressibleFunction.ForwardKernelName);
                compressibleFunction.BackwardgWKernel = program.CreateKernel(compressibleFunction.BackwardgWKernelName);
                compressibleFunction.BackwardgXKernel = program.CreateKernel(compressibleFunction.BackwardgXKernelName);
            }
        }
    }
}
