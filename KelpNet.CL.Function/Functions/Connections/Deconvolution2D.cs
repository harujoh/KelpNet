using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
using Deconvolution2DFunc = KelpNet.CPU.Deconvolution2DD;
#else
using KelpNet.CL.Properties;
using Real = System.Single;
using Deconvolution2DFunc = KelpNet.CPU.Deconvolution2DF;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "Deconvolution2D", Namespace = "KelpNet")]
    public class Deconvolution2D<T> : CPU.Deconvolution2D<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Deconvolution2D";
        public string KernelSource { get; set; }

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

        [DataMember]
        public new ICompressibleActivation<T> Activation { get; set; }

        public bool SetParallel(bool enable)
        {
            KernelSource = OpenCL.GetKernelSource(Resources.Deconvolution2D);
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
            this.PadX = deconv2D.PadX;
            this.PadY = deconv2D.PadY;
            this.StrideX = deconv2D.StrideX;
            this.StrideY = deconv2D.StrideY;

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
                        deconvolution2DF.SingleInputForward = x => Deconvolution2DF.SingleInputForward(x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF.ForwardKernel, deconvolution2DF);
                        deconvolution2DF.SingleOutputBackward = (y, x) => Deconvolution2DF.SingleOutputBackward(y, x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF.BackwardgWKernel, deconvolution2DF.BackwardgXKernel, deconvolution2DF.Activation);
                        break;

                    case Deconvolution2D<double> deconvolution2DD:
                        deconvolution2DD.SingleInputForward = x => Deconvolution2DD.SingleInputForward(x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD.ForwardKernel, deconvolution2DD);
                        deconvolution2DD.SingleOutputBackward = (y, x) => Deconvolution2DD.SingleOutputBackward(y, x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD.BackwardgWKernel, deconvolution2DD.BackwardgXKernel, deconvolution2DD.Activation);
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
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ComputeKernel forwardKernel, IFunction<Real> deconv2d)
        {
            int inputCount = weight.Shape[0];
            int outputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            int outputHeight = (input.Shape[1] - 1) * strideY + kernelHeight - padY * 2;
            int outputWidth = (input.Shape[2] - 1) * strideX + kernelWidth - padX * 2;

            Real[] result = new Real[input.BatchCount * outputCount * outputWidth * outputHeight];

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

            return NdArray.Convert(result, new[] { outputCount, outputHeight, outputWidth }, input.BatchCount, deconv2d);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ComputeKernel backwardgWKernel, ComputeKernel backwardgXKernel, ICompressibleActivation<Real> activation)
        {
            int inputCount = weight.Shape[0];
            int outputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y, x) : y.Grad;
            if (bias != null)
            {
                Deconvolution2DFunc.CalcBiasGrad(activatedgy, bias.Grad, y.Shape, y.BatchCount);
            }

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
                    backwardgWKernel.SetValueArgument(4, outputCount);
                    backwardgWKernel.SetValueArgument(5, y.Length);
                    backwardgWKernel.SetValueArgument(6, y.Shape[1]);
                    backwardgWKernel.SetValueArgument(7, y.Shape[2]);
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
                    backwardgXKernel.SetMemoryArgument(0, gpugY);
                    backwardgXKernel.SetMemoryArgument(1, gpuW);
                    backwardgXKernel.SetMemoryArgument(2, gpugX);
                    backwardgXKernel.SetValueArgument(3, outputCount);
                    backwardgXKernel.SetValueArgument(4, inputCount);
                    backwardgXKernel.SetValueArgument(5, y.Length);
                    backwardgXKernel.SetValueArgument(6, y.Shape[1]);
                    backwardgXKernel.SetValueArgument(7, y.Shape[2]);
                    backwardgXKernel.SetValueArgument(8, x.Shape[1]);
                    backwardgXKernel.SetValueArgument(9, x.Shape[2]);
                    backwardgXKernel.SetValueArgument(10, x.Length);
                    backwardgXKernel.SetValueArgument(11, strideX);
                    backwardgXKernel.SetValueArgument(12, strideY);
                    backwardgXKernel.SetValueArgument(13, padX);
                    backwardgXKernel.SetValueArgument(14, padY);
                    backwardgXKernel.SetValueArgument(15, kernelHeight);
                    backwardgXKernel.SetValueArgument(16, kernelWidth);

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
