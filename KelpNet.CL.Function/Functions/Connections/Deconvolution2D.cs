using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
using KelpNet.CL.Properties;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "Deconvolution2D", Namespace = "KelpNet")]
    public class Deconvolution2D<T> : CPU.Deconvolution2D<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Deconvolution2D";
        public string KernelSource { get; set; } = OpenCL.GetKernelSource(Resources.Deconvolution2D);

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

        public bool SetParallel(bool enable)
        {
            bool result = this.SetParallel<T>(enable);
            this.InitFunc(new StreamingContext());
            return result;
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = "Deconvolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int[] kSize, int[] subSample = null, int[] trim = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = "Deconvolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kSize, subSample, trim, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Deconvolution2D(CPU.Deconvolution2D<T> deconv2D) : base(deconv2D.Name, deconv2D.InputNames, deconv2D.OutputNames)
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

            this.Activation = (ICompressibleActivation<T>)CLConverter.Convert(deconv2D.Activation);

            this.SetParallel(true);
            this.InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        public void InitFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case Deconvolution2D<float> deconvolution2DF:
                        deconvolution2DF.SingleInputForward = x => Deconvolution2DF.SingleInputForward(x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.NoBias, deconvolution2DF.InputCount, deconvolution2DF.OutputCount, deconvolution2DF.KernelWidth, deconvolution2DF.KernelHeight, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF.ForwardKernel, deconvolution2DF);
                        deconvolution2DF.SingleOutputBackward = (y, x) => Deconvolution2DF.SingleOutputBackward(y, x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.NoBias, deconvolution2DF.InputCount, deconvolution2DF.OutputCount, deconvolution2DF.KernelWidth, deconvolution2DF.KernelHeight, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF.BackwardgWKernel, deconvolution2DF.BackwardgXKernel, CPU.Deconvolution2DF.CalcBiasGrad, deconvolution2DF.Activation);
                        break;

                    case Deconvolution2D<double> deconvolution2DD:
                        deconvolution2DD.SingleInputForward = x => Deconvolution2DD.SingleInputForward(x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.NoBias, deconvolution2DD.InputCount, deconvolution2DD.OutputCount, deconvolution2DD.KernelWidth, deconvolution2DD.KernelHeight, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD.ForwardKernel, deconvolution2DD);
                        deconvolution2DD.SingleOutputBackward = (y, x) => Deconvolution2DD.SingleOutputBackward(y, x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.NoBias, deconvolution2DD.InputCount, deconvolution2DD.OutputCount, deconvolution2DD.KernelWidth, deconvolution2DD.KernelHeight, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD.BackwardgWKernel, deconvolution2DD.BackwardgXKernel, CPU.Deconvolution2DD.CalcBiasGrad, deconvolution2DD.Activation);
                        break;
                }
            }
        }
    }
#endif

#if DOUBLE
    public static class Deconvolution2DD
#else
    public static class Deconvolution2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int inputCount, int outputCount, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, ComputeKernel ForwardKernel, IFunction<Real> deconv2d)
        {
            int outputHeight = (input.Shape[1] - 1) * strideY + kernelHeight - padY * 2;
            int outputWidth = (input.Shape[2] - 1) * strideX + kernelWidth - padX * 2;

            Real[] result = new Real[input.BatchCount * outputCount * outputWidth * outputHeight];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, noBias ? new Real[outputCount] : bias.Data))
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
                ForwardKernel.SetValueArgument(9, strideX);
                ForwardKernel.SetValueArgument(10, strideY);
                ForwardKernel.SetValueArgument(11, padX);
                ForwardKernel.SetValueArgument(12, padY);
                ForwardKernel.SetValueArgument(13, kernelHeight);
                ForwardKernel.SetValueArgument(14, kernelWidth);
                ForwardKernel.SetValueArgument(15, outputCount);
                ForwardKernel.SetValueArgument(16, inputCount);

                OpenCL.CommandQueue.Execute
                    (
                        ForwardKernel,
                        null,
                        new long[] { input.BatchCount * outputCount, outputHeight, outputWidth },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, new[] { outputCount, outputHeight, outputWidth }, input.BatchCount, deconv2d);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int inputCount, int outputCount, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, ComputeKernel BackwardgWKernel, ComputeKernel BackwardgXKernel, Action<Real[], Real[], int[], int> CalcBiasGrad, KelpNet.ICompressibleActivation<Real> Activation)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = Activation != null ? Activation.GetActivatedgy(y) : y.Grad;
            if (!noBias)
            {
                CalcBiasGrad(activatedgy, bias.Grad, y.Shape, y.BatchCount);
            }

            //gyは共通で使用
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
                {
                    BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgWKernel.SetValueArgument(4, outputCount);
                    BackwardgWKernel.SetValueArgument(5, y.Length);
                    BackwardgWKernel.SetValueArgument(6, y.Shape[1]);
                    BackwardgWKernel.SetValueArgument(7, y.Shape[2]);
                    BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                    BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                    BackwardgWKernel.SetValueArgument(10, x.Length);
                    BackwardgWKernel.SetValueArgument(11, strideX);
                    BackwardgWKernel.SetValueArgument(12, strideY);
                    BackwardgWKernel.SetValueArgument(13, padX);
                    BackwardgWKernel.SetValueArgument(14, padY);
                    BackwardgWKernel.SetValueArgument(15, kernelHeight);
                    BackwardgWKernel.SetValueArgument(16, kernelWidth);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgWKernel,
                        null,
                        new long[] { inputCount * outputCount, kernelHeight, kernelWidth },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugW, ref weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, weight.Data))
                {
                    BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    BackwardgXKernel.SetValueArgument(3, outputCount);
                    BackwardgXKernel.SetValueArgument(4, inputCount);
                    BackwardgXKernel.SetValueArgument(5, y.Length);
                    BackwardgXKernel.SetValueArgument(6, y.Shape[1]);
                    BackwardgXKernel.SetValueArgument(7, y.Shape[2]);
                    BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                    BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                    BackwardgXKernel.SetValueArgument(10, x.Length);
                    BackwardgXKernel.SetValueArgument(11, strideX);
                    BackwardgXKernel.SetValueArgument(12, strideY);
                    BackwardgXKernel.SetValueArgument(13, padX);
                    BackwardgXKernel.SetValueArgument(14, padY);
                    BackwardgXKernel.SetValueArgument(15, kernelHeight);
                    BackwardgXKernel.SetValueArgument(16, kernelWidth);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgXKernel,
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
