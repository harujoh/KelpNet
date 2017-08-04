using System;
using System.Linq;
using Cloo;
using KelpNet.Common.Functions;

namespace KelpNet.Common.Activations
{
    [Serializable]
    public abstract class Activation : NeedPreviousOutputFunction
    {
        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardKernel;

        //GPU向けのActivate関数の文字列
        public abstract string ForwardActivateFunctionString { get; }
        public abstract string BackwardActivateFunctionString { get; }

        //.Netで使用するActivateの仮想関数
        public abstract void ForwardActivate(ref Real x);
        public abstract void BackwardActivate(ref Real gy, Real y);

        public string ForwardKernelName { get; }
        public string BackwardKernelName { get; }

        protected Activation(string name, bool isGpu) : base(name, isGpu)
        {
            string kernelNameBase = name.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardKernelName = kernelNameBase + "Backward";
            string kernaelHeader = "__kernel void " + kernelNameBase;

            this.ForwardActivateKernelString = kernaelHeader + this.ForwardActivateKernelString;
            this.BackwardActivateKernelString = kernaelHeader + this.BackwardActivateKernelString;
        }

        protected string ForwardActivateKernelString =
@"Forward(__global Real *gpuY)
{
	int i = get_global_id(0);

    ForwardActivate(gpuY + i);
}";

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
                    ForwardKernel.SetMemoryArgument(0, gpuY);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
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

        protected string BackwardActivateKernelString =
@"Backward(__global read_only Real *gpuY,
           __global Real *gpugX)
{
	int i = get_global_id(0);

    Real gpugY = gpugX[i];
    BackwardActivate(gpuY[i], &gpugY);
    
    gpugX[i] = gpugY;
}";

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
                    BackwardKernel.SetMemoryArgument(0, gpuY);
                    BackwardKernel.SetMemoryArgument(1, gpugX);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
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
