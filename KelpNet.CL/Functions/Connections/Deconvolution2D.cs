using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "Deconvolution2D", Namespace = "KelpNet")]
    public class Deconvolution2D : CPU.Deconvolution2D, ICompressibleFunction
    {
        public string FunctionName => "Deconvolution2D";
        public string KernelSource => OpenCL.GetKernelSource(Resources.Deconvolution2D);

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgWKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgXKernel { get; set; }

        [DataMember]
        public bool IsParallel { get; set; }

        [DataMember]
        public string ForwardKernelName { get; set; }

        [DataMember]
        public string BackwardgWKernelName { get; set; }

        [DataMember]
        public string BackwardgXKernelName { get; set; }


        bool IParallelizable.SetParallel(bool enable)
        {
            return this.SetParallel(enable);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = "Deconvolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int[] kSize, int[] subSample = null, int[] trim = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = "Deconvolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kSize, subSample, trim, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Deconvolution2D(CPU.Deconvolution2D deconv2D) : base(deconv2D.Name, deconv2D.InputNames, deconv2D.OutputNames)
        {
            this.KernelWidth = deconv2D.KernelWidth;
            this.KernelHeight = deconv2D.KernelHeight;
            this.PadX = deconv2D.PadX;
            this.PadY = deconv2D.PadY;
            this.StrideX = deconv2D.StrideX;
            this.StrideY = deconv2D.StrideY;
            this.NoBias = deconv2D.NoBias;

            this.OutputCount = deconv2D.OutputCount;
            this.InputCount = deconv2D.InputCount;

            this.Weight = deconv2D.Weight;
            this.Bias = deconv2D.Bias;

            this.Parameters = deconv2D.Parameters;

            this.Activation = (ICompressibleActivation)CLConverter.Convert(deconv2D.Activation);

            this.SetParallel(true);
        }

        public override NdArray SingleInputForward(NdArray input)
        {
            //フラグチェック
            if (!IsParallel) return base.SingleInputForward(input);

            int outputHeight = (input.Shape[1] - 1) * this.StrideY + this.KernelHeight - this.PadY * 2;
            int outputWidth = (input.Shape[2] - 1) * this.StrideX + this.KernelWidth - this.PadX * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

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
            if (!IsParallel)
            {
                base.SingleOutputBackward(y, x);
                return;
            }

            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = this.Activation != null ? this.GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

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
                    this.BackwardgWKernel.SetValueArgument(4, this.OutputCount);
                    this.BackwardgWKernel.SetValueArgument(5, y.Length);
                    this.BackwardgWKernel.SetValueArgument(6, y.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(7, y.Shape[2]);
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
                        new long[] { this.InputCount * this.OutputCount, this.KernelHeight, this.KernelWidth },
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
                    this.BackwardgXKernel.SetValueArgument(3, this.OutputCount);
                    this.BackwardgXKernel.SetValueArgument(4, this.InputCount);
                    this.BackwardgXKernel.SetValueArgument(5, y.Length);
                    this.BackwardgXKernel.SetValueArgument(6, y.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(7, y.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(10, x.Length);
                    this.BackwardgXKernel.SetValueArgument(11, this.StrideX);
                    this.BackwardgXKernel.SetValueArgument(12, this.StrideY);
                    this.BackwardgXKernel.SetValueArgument(13, this.PadX);
                    this.BackwardgXKernel.SetValueArgument(14, this.PadY);
                    this.BackwardgXKernel.SetValueArgument(15, this.KernelHeight);
                    this.BackwardgXKernel.SetValueArgument(16, this.KernelWidth);

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
