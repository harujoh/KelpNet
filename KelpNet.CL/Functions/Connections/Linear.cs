using System;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class Linear : CPU.Linear, ICompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        public ComputeKernel ForwardKernel { get; set; }
        public ComputeKernel BackwardgWKernel { get; set; }
        public ComputeKernel BackwardgXKernel { get; set; }
        public string ForwardKernelName { get; set; }
        public string BackwardgWKernelName { get; set; }
        public string BackwardgXKernelName { get; set; }
        public string KernelString { get; set; }
        public bool IsParallel { get; set; }

        void IParallelizable.InitParallel()
        {
            this.InitParallel();
        }

        bool IParallelizable.SetParallel(bool enable)
        {
            return this.SetParallel(enable);
        }

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputCount, outputCount, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.Initialize(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Linear), activation, name, inputNames, outputNames, gpuEnable);
        }

        public NdArray NeedPreviousForwardGpu(NdArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, y))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpuY);
                ForwardKernel.SetValueArgument(3, this.OutputCount);
                ForwardKernel.SetValueArgument(4, this.InputCount);

                OpenCL.CommandQueue.Execute
                    (
                        ForwardKernel,
                        null,
                        new long[] { OutputCount, x.BatchCount },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgY.Length; i++)
            {
                activatedgY[i] = this.Activation.BackwardActivate(y.Grad[i], y.Data[i]);
            }

            return activatedgY;
        }


        public void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = this.Activation != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
                {
                    BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgWKernel.SetValueArgument(4, this.OutputCount);
                    BackwardgWKernel.SetValueArgument(5, this.InputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgWKernel,
                        null,
                        new long[] { this.InputCount, this.OutputCount },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
                {
                    BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    BackwardgXKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgXKernel.SetValueArgument(4, this.OutputCount);
                    BackwardgXKernel.SetValueArgument(5, this.InputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgXKernel,
                        null,
                        new long[] { this.InputCount, y.BatchCount },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
