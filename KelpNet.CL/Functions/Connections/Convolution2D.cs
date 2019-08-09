using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "Convolution2D", Namespace = "KelpNet")]
    public class Convolution2D : CPU.Convolution2D, ICompressibleFunction
    {
        public string FunctionName => "Convolution2D";
        public string KernelSource => OpenCL.GetKernelSource(Resources.Convolution2D);

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgWKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgXKernel { get; set; }

        [DataMember]
        public string ForwardKernelName { get; set; }

        [DataMember]
        public string BackwardgWKernelName { get; set; }

        [DataMember]
        public string BackwardgXKernelName { get; set; }

        [DataMember]
        public bool IsParallel { get; set; }

        bool IParallelizable.SetParallel(bool enable)
        {
            return this.SetParallel(enable);
        }

        public Convolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = "Convolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = "Convolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Convolution2D(Linear linear) : base(linear)
        {
            this.SetParallel(linear.IsParallel);
        }

        //Convert
        public Convolution2D(CPU.Convolution2D conv2d) : base(conv2d.Name, conv2d.InputNames, conv2d.OutputNames)
        {
            this.KernelWidth = conv2d.KernelWidth;
            this.KernelHeight = conv2d.KernelHeight;
            this.StrideX = conv2d.StrideX;
            this.StrideY = conv2d.StrideY;
            this.PadX = conv2d.PadX;
            this.PadY = conv2d.PadY;
            this.NoBias = conv2d.NoBias;

            this.OutputCount = conv2d.OutputCount;
            this.InputCount = conv2d.InputCount;

            this.Weight = conv2d.Weight;
            this.Bias = conv2d.Bias;

            this.Parameters = conv2d.Parameters;

            this.Activation = (ICompressibleActivation)CLConverter.Convert(conv2d.Activation);

            this.SetParallel(true);
        }

        public override NdArray SingleInputForward(NdArray input)
        {
            //フラグチェック
            if (!IsParallel) return base.SingleInputForward(input);

            int outputHeight = (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.NoBias ? new Real[OutputCount] : this.Bias.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpub);
                ForwardKernel.SetMemoryArgument(3, gpuY);
                ForwardKernel.SetValueArgument(4, input.Shape[1]);
                ForwardKernel.SetValueArgument(5, input.Shape[2]);
                ForwardKernel.SetValueArgument(6, input.Length);
                ForwardKernel.SetValueArgument(7, outputWidth);
                ForwardKernel.SetValueArgument(8, outputHeight);
                ForwardKernel.SetValueArgument(9, this.StrideX);
                ForwardKernel.SetValueArgument(10, this.StrideY);
                ForwardKernel.SetValueArgument(11, this.PadX);
                ForwardKernel.SetValueArgument(12, this.PadY);
                ForwardKernel.SetValueArgument(13, this.KernelHeight);
                ForwardKernel.SetValueArgument(14, this.KernelWidth);
                ForwardKernel.SetValueArgument(15, this.OutputCount);
                ForwardKernel.SetValueArgument(16, this.InputCount);

                OpenCL.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            //フラグチェック
            if (!IsParallel)
            {
                base.SingleOutputBackward(y, x);
                return;
            }

            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = this.Activation != null ? this.GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            int kyStartPrevOffset = KernelHeight - PadY - x.Shape[1];
            int kxStartPrevOffset = KernelWidth - PadX - x.Shape[2];

            //gyは共通で使用
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
                {
                    this.BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    this.BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    this.BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    this.BackwardgWKernel.SetValueArgument(4, this.InputCount);
                    this.BackwardgWKernel.SetValueArgument(5, y.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(6, y.Shape[2]);
                    this.BackwardgWKernel.SetValueArgument(7, y.Length);
                    this.BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                    this.BackwardgWKernel.SetValueArgument(10, x.Length);
                    this.BackwardgWKernel.SetValueArgument(11, this.StrideX);
                    this.BackwardgWKernel.SetValueArgument(12, this.StrideY);
                    this.BackwardgWKernel.SetValueArgument(13, this.PadX);
                    this.BackwardgWKernel.SetValueArgument(14, this.PadY);
                    this.BackwardgWKernel.SetValueArgument(15, this.KernelHeight);
                    this.BackwardgWKernel.SetValueArgument(16, this.KernelWidth);

                    OpenCL.CommandQueue.Execute
                    (
                        this.BackwardgWKernel,
                        null,
                        new long[] { OutputCount * InputCount, this.KernelHeight, this.KernelWidth },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
                {
                    this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    this.BackwardgXKernel.SetValueArgument(3, y.Length);
                    this.BackwardgXKernel.SetValueArgument(4, y.Shape[0]);
                    this.BackwardgXKernel.SetValueArgument(5, y.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(6, y.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(7, x.Length);
                    this.BackwardgXKernel.SetValueArgument(8, x.Shape[0]);
                    this.BackwardgXKernel.SetValueArgument(9, x.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(10, x.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(11, this.StrideX);
                    this.BackwardgXKernel.SetValueArgument(12, this.StrideY);
                    this.BackwardgXKernel.SetValueArgument(13, this.PadX);
                    this.BackwardgXKernel.SetValueArgument(14, this.PadY);
                    this.BackwardgXKernel.SetValueArgument(15, this.KernelWidth);
                    this.BackwardgXKernel.SetValueArgument(16, this.KernelHeight);
                    this.BackwardgXKernel.SetValueArgument(17, kxStartPrevOffset);
                    this.BackwardgXKernel.SetValueArgument(18, kyStartPrevOffset);

                    OpenCL.CommandQueue.Execute
                    (
                        this.BackwardgXKernel,
                        null,
                        new long[] { x.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
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
