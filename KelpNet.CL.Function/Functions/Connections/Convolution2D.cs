using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
using Convolution2DFunc = KelpNet.CPU.Convolution2DD;
#elif NETSTANDARD2_1
using KelpNet.CL.Properties;
using Real = System.Single;
using Math = System.MathF;
using Convolution2DFunc = KelpNet.CPU.Convolution2DF;
#else
using KelpNet.CL.Properties;
using Real = System.Single;
using Math = KelpNet.MathF;
using Convolution2DFunc = KelpNet.CPU.Convolution2DF;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "Convolution2D", Namespace = "KelpNet")]
    public class Convolution2D<T> : CPU.Convolution2D<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Convolution2D";
        public string KernelSource { get; set; }

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

        [DataMember]
        public new ICompressibleActivation<T> Activation { get; set; }

        public bool SetParallel(bool enable)
        {
            KernelSource = OpenCL.GetKernelSource(Resources.Convolution2D);
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
            this.StrideX = conv2d.StrideX;
            this.StrideY = conv2d.StrideY;
            this.PadX = conv2d.PadX;
            this.PadY = conv2d.PadY;

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
                        convolution2DF.SingleInputForward = x => Convolution2DF.SingleInputForward(x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.ForwardKernel, convolution2DF);
                        convolution2DF.SingleOutputBackward = (y, x) => Convolution2DF.SingleOutputBackward(y, x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.BackwardgWKernel, convolution2DF.BackwardgXKernel, convolution2DF.Activation);
                        break;

                    case Convolution2D<double> convolution2DD:
                        convolution2DD.SingleInputForward = x => Convolution2DD.SingleInputForward(x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.ForwardKernel, convolution2DD);
                        convolution2DD.SingleOutputBackward = (y, x) => Convolution2DD.SingleOutputBackward(y, x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.BackwardgWKernel, convolution2DD.BackwardgXKernel, convolution2DD.Activation);
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
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ComputeKernel forwardKernel, IFunction<Real> conv2d)
        {
            int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            int outputHeight = (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0f) / strideX) + 1;

            Real[] result = new Real[outputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, bias == null ? new Real[outputCount] : bias.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                forwardKernel.SetMemoryArgument(0, gpuX);
                forwardKernel.SetMemoryArgument(1, gpuW);
                forwardKernel.SetMemoryArgument(2, gpub);
                forwardKernel.SetMemoryArgument(3, gpuY);
                forwardKernel.SetValueArgument(4, input.Shape[1]);
                forwardKernel.SetValueArgument(5, input.Shape[2]);
                forwardKernel.SetValueArgument(6, input.Length);
                forwardKernel.SetValueArgument(7, outputWidth);
                forwardKernel.SetValueArgument(8, outputHeight);
                forwardKernel.SetValueArgument(9, strideX);
                forwardKernel.SetValueArgument(10, strideY);
                forwardKernel.SetValueArgument(11, padX);
                forwardKernel.SetValueArgument(12, padY);
                forwardKernel.SetValueArgument(13, kernelHeight);
                forwardKernel.SetValueArgument(14, kernelWidth);
                forwardKernel.SetValueArgument(15, outputCount);
                forwardKernel.SetValueArgument(16, inputCount);

                OpenCL.CommandQueue.Execute
                (
                    forwardKernel,
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

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ComputeKernel backwardgWKernel, ComputeKernel backwardgXKernel, ICompressibleActivation<Real> activation)
        {
            int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y, x) : y.Grad;

            if (bias != null)
            {
                Convolution2DFunc.CalcBiasGrad(activatedgy, y.Shape, y.BatchCount, bias.Grad);
            }

            int kyStartPrevOffset = kernelHeight - padY - x.Shape[1];
            int kxStartPrevOffset = kernelWidth - padX - x.Shape[2];

            //gyは共通で使用
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
                {
                    backwardgWKernel.SetMemoryArgument(0, gpugY);
                    backwardgWKernel.SetMemoryArgument(1, gpuX);
                    backwardgWKernel.SetMemoryArgument(2, gpugW);
                    backwardgWKernel.SetValueArgument(3, y.BatchCount);
                    backwardgWKernel.SetValueArgument(4, inputCount);
                    backwardgWKernel.SetValueArgument(5, y.Shape[1]);
                    backwardgWKernel.SetValueArgument(6, y.Shape[2]);
                    backwardgWKernel.SetValueArgument(7, y.Length);
                    backwardgWKernel.SetValueArgument(8, x.Shape[1]);
                    backwardgWKernel.SetValueArgument(9, x.Shape[2]);
                    backwardgWKernel.SetValueArgument(10, x.Length);
                    backwardgWKernel.SetValueArgument(11, strideX);
                    backwardgWKernel.SetValueArgument(12, strideY);
                    backwardgWKernel.SetValueArgument(13, padX);
                    backwardgWKernel.SetValueArgument(14, padY);
                    backwardgWKernel.SetValueArgument(15, kernelHeight);
                    backwardgWKernel.SetValueArgument(16, kernelWidth);

                    OpenCL.CommandQueue.Execute
                    (
                        backwardgWKernel,
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
                    backwardgXKernel.SetMemoryArgument(0, gpugY);
                    backwardgXKernel.SetMemoryArgument(1, gpuW);
                    backwardgXKernel.SetMemoryArgument(2, gpugX);
                    backwardgXKernel.SetValueArgument(3, y.Length);
                    backwardgXKernel.SetValueArgument(4, y.Shape[0]);
                    backwardgXKernel.SetValueArgument(5, y.Shape[1]);
                    backwardgXKernel.SetValueArgument(6, y.Shape[2]);
                    backwardgXKernel.SetValueArgument(7, x.Length);
                    backwardgXKernel.SetValueArgument(8, x.Shape[0]);
                    backwardgXKernel.SetValueArgument(9, x.Shape[1]);
                    backwardgXKernel.SetValueArgument(10, x.Shape[2]);
                    backwardgXKernel.SetValueArgument(11, strideX);
                    backwardgXKernel.SetValueArgument(12, strideY);
                    backwardgXKernel.SetValueArgument(13, padX);
                    backwardgXKernel.SetValueArgument(14, padY);
                    backwardgXKernel.SetValueArgument(15, kernelWidth);
                    backwardgXKernel.SetValueArgument(16, kernelHeight);
                    backwardgXKernel.SetValueArgument(17, kxStartPrevOffset);
                    backwardgXKernel.SetValueArgument(18, kyStartPrevOffset);

                    OpenCL.CommandQueue.Execute
                    (
                        backwardgXKernel,
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
