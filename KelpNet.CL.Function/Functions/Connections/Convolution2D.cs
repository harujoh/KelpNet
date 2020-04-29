using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
using KelpNet.CL.Properties;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "Convolution2D", Namespace = "KelpNet")]
    public class Convolution2D<T> : CPU.Convolution2D<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Convolution2D";
        public string KernelSource { get; set; } = OpenCL.GetKernelSource(Resources.Convolution2D);

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

        public bool SetParallel(bool enable)
        {
            bool result = this.SetParallel<T>(enable);
            this.InitFunc(new StreamingContext());
            return result;
        }

        public Convolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = "Convolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = "Convolution2D", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputChannels, outputChannels, kernelSize, stride, pad, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Convolution2D(Linear<T> linear) : base(linear)
        {
            this.SetParallel(linear.IsParallel);
            this.InitFunc(new StreamingContext());
        }

        //Convert
        public Convolution2D(CPU.Convolution2D<T> conv2d) : base(conv2d.Name, conv2d.InputNames, conv2d.OutputNames)
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

            this.Activation = (ICompressibleActivation<T>)CLConverter.Convert(conv2d.Activation);

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
                    case Convolution2D<float> convolution2DF:
                        convolution2DF.SingleInputForward = x => Convolution2DF.SingleInputForward(x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.NoBias, convolution2DF.InputCount, convolution2DF.OutputCount, convolution2DF.KernelWidth, convolution2DF.KernelHeight, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.ForwardKernel, convolution2DF);
                        convolution2DF.SingleOutputBackward = (y, x) => Convolution2DF.SingleOutputBackward(y, x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.NoBias, convolution2DF.InputCount, convolution2DF.OutputCount, convolution2DF.KernelWidth, convolution2DF.KernelHeight, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.BackwardgWKernel, convolution2DF.BackwardgXKernel, CPU.Convolution2DF.CalcBiasGrad, convolution2DF.Activation);
                        break;

                    case Convolution2D<double> convolution2DD:
                        convolution2DD.SingleInputForward = x => Convolution2DD.SingleInputForward(x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.NoBias, convolution2DD.InputCount, convolution2DD.OutputCount, convolution2DD.KernelWidth, convolution2DD.KernelHeight, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.ForwardKernel, convolution2DD);
                        convolution2DD.SingleOutputBackward = (y, x) => Convolution2DD.SingleOutputBackward(y, x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.NoBias, convolution2DD.InputCount, convolution2DD.OutputCount, convolution2DD.KernelWidth, convolution2DD.KernelHeight, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.BackwardgWKernel, convolution2DD.BackwardgXKernel, CPU.Convolution2DD.CalcBiasGrad, convolution2DD.Activation);
                        break;
                }
            }
        }
    }
#endif

#if DOUBLE
    public static class Convolution2DD
#else
    public static class Convolution2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int inputCount, int outputCount, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, ComputeKernel ForwardKernel, IFunction<Real> conv2d)
        {
            int outputHeight = (int)KelpMath.Floor((input.Shape[1] - kernelHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = (int)KelpMath.Floor((input.Shape[2] - kernelWidth + padX * 2.0f) / strideX) + 1;

            Real[] result = new Real[outputCount * outputHeight * outputWidth * input.BatchCount];

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

            return NdArray.Convert(result, new[] { outputCount, outputHeight, outputWidth }, input.BatchCount, conv2d);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int inputCount, int outputCount, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, ComputeKernel BackwardgWKernel, ComputeKernel BackwardgXKernel, Action<Real[], int[], int, Real[]> CalcBiasGrad, KelpNet.ICompressibleActivation<Real> Activation)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = Activation != null ? Activation.GetActivatedgy(y) : y.Grad;

            if (!noBias)
            {
                CalcBiasGrad(activatedgy, y.Shape, y.BatchCount, bias.Grad);
            }

            int kyStartPrevOffset = kernelHeight - padY - x.Shape[1];
            int kxStartPrevOffset = kernelWidth - padX - x.Shape[2];

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
                    BackwardgWKernel.SetValueArgument(4, inputCount);
                    BackwardgWKernel.SetValueArgument(5, y.Shape[1]);
                    BackwardgWKernel.SetValueArgument(6, y.Shape[2]);
                    BackwardgWKernel.SetValueArgument(7, y.Length);
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
                        new long[] { outputCount * inputCount, kernelHeight, kernelWidth },
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
                    BackwardgXKernel.SetValueArgument(3, y.Length);
                    BackwardgXKernel.SetValueArgument(4, y.Shape[0]);
                    BackwardgXKernel.SetValueArgument(5, y.Shape[1]);
                    BackwardgXKernel.SetValueArgument(6, y.Shape[2]);
                    BackwardgXKernel.SetValueArgument(7, x.Length);
                    BackwardgXKernel.SetValueArgument(8, x.Shape[0]);
                    BackwardgXKernel.SetValueArgument(9, x.Shape[1]);
                    BackwardgXKernel.SetValueArgument(10, x.Shape[2]);
                    BackwardgXKernel.SetValueArgument(11, strideX);
                    BackwardgXKernel.SetValueArgument(12, strideY);
                    BackwardgXKernel.SetValueArgument(13, padX);
                    BackwardgXKernel.SetValueArgument(14, padY);
                    BackwardgXKernel.SetValueArgument(15, kernelWidth);
                    BackwardgXKernel.SetValueArgument(16, kernelHeight);
                    BackwardgXKernel.SetValueArgument(17, kxStartPrevOffset);
                    BackwardgXKernel.SetValueArgument(18, kyStartPrevOffset);

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
