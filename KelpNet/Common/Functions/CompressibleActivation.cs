using System;
using System.Linq;
using Cloo;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleActivation : NeedPreviousOutputFunction
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

        protected CompressibleActivation(string name) : base(name)
        {
            string kernelNameBase = name.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardKernelName = kernelNameBase + "Backward";

            this.ActivateKernelString = Weaver.GetKernelSource(FUNCTION_NAME).Replace("/*kernelNameBase*/", kernelNameBase);
        }

        protected override void CreateKernel()
        {
            string kernelSource = this.ActivateFunctionString + ActivateKernelString;

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
            this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            Real[] y = x.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    this.ForwardActivate(ref y[i]);
                }
            }
            else
            {
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
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = gy.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < gx.Length; i++)
                {
                    this.BackwardActivate(ref gx[i], prevOutput.Data[i]);
                }
            }
            else
            {
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
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
