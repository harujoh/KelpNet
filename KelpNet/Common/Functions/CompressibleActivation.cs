using System;
using System.Collections.Generic;
using System.Linq;
using Cloo;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleActivation : NeedPreviousOutputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Activation";

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardKernel;

        //GPU向けのActivate関数の文字列
        public string ActivateFunctionString;

        //.Netで使用するActivateの仮想関数
        public abstract void ForwardActivate(ref Real x);
        public abstract void BackwardActivate(ref Real gy, Real y);

        public string ForwardKernelName { get; }
        public string BackwardKernelName { get; }

        protected string ActivateKernelString;

        protected CompressibleActivation(string name, bool gpuEnable, string functionName, params KeyValuePair<string, string>[] parameters) : base(name)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardKernelName = kernelNameBase + "Backward";

            this.ActivateKernelString = Weaver.GetKernelSource(FUNCTION_NAME).Replace("/*kernelNameBase*/", kernelNameBase);
            this.ActivateFunctionString = Weaver.GetKernelSource(functionName);

            foreach (var parameter in parameters)
            {
                this.ActivateFunctionString = this.ActivateFunctionString.Replace(parameter.Key, parameter.Value);
            }

            this.SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            if (this.GpuEnable)
            {
                CreateKernel();
                NeedPreviousForward = NeedPreviousForwardGpu;
                NeedPreviousBackward = NeedPreviousBackwardGpu;
            }
            else
            {
                NeedPreviousForward = NeedPreviousForwardCpu;
                NeedPreviousBackward = NeedPreviousBackwardCpu;
            }

            return GpuEnable;
        }

        public void CreateKernel()
        {
            string kernelSource = this.ActivateFunctionString + this.ActivateKernelString;

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
            this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
        }

        protected BatchArray NeedPreviousForwardCpu(BatchArray x)
        {
            Real[] y = x.Data.ToArray();

            for (int i = 0; i < y.Length; i++)
            {
                this.ForwardActivate(ref y[i]);
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected BatchArray NeedPreviousForwardGpu(BatchArray x)
        {
            Real[] y = x.Data.ToArray();

            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
            {
                this.ForwardKernel.SetMemoryArgument(0, gpuY);

                Weaver.CommandQueue.Execute
                    (
                        this.ForwardKernel,
                        null,
                        new long[] { x.Data.Length },
                        null,
                        null
                    );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected BatchArray NeedPreviousBackwardCpu(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = gy.Data.ToArray();

            for (int i = 0; i < gx.Length; i++)
            {
                this.BackwardActivate(ref gx[i], prevOutput.Data[i]);
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }

        protected BatchArray NeedPreviousBackwardGpu(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = gy.Data.ToArray();

            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutput.Data))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gx))
            {
                this.BackwardKernel.SetMemoryArgument(0, gpuY);
                this.BackwardKernel.SetMemoryArgument(1, gpugX);

                Weaver.CommandQueue.Execute
                    (
                        this.BackwardKernel,
                        null,
                        new long[] { gy.Data.Length },
                        null,
                        null
                    );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
