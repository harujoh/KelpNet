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
    [DataContract(Name = "Linear", Namespace = "KelpNet")]
    public class Linear<T> : CPU.Linear<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Linear";
        public string KernelSource { get; set; } = OpenCL.GetKernelSource(Resources.Linear);

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

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, T[] initialb = null, ICompressibleActivation<T> activation = null, string name = "Linear", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputCount, outputCount, noBias, initialW, initialb, activation, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Linear(CPU.Linear<T> linear) : base(linear.Name, linear.InputNames, linear.OutputNames)
        {
            this.OutputCount = linear.OutputCount;
            this.InputCount = linear.InputCount;

            this.NoBias = linear.NoBias;

            this.Weight = linear.Weight;
            this.Bias = linear.Bias;

            this.Parameters = linear.Parameters;

            this.Activation = (ICompressibleActivation<T>)CLConverter.Convert(linear.Activation);

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
                    case Linear<float> linearF:
                        linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Weight.Data, linearF.Bias.Data, linearF.NoBias, linearF.InputCount, linearF.OutputCount, linearF.ForwardKernel, CPU.LinearF.GetBiasedValue, linearF);
                        linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Weight, linearF.Bias.Grad, linearF.NoBias, linearF.InputCount, linearF.OutputCount, linearF.BackwardgWKernel, linearF.BackwardgXKernel, CPU.LinearF.CalcBiasGrad, linearF.Activation);
                        break;

                    case Linear<double> linearD:
                        linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Weight.Data, linearD.Bias.Data, linearD.NoBias, linearD.InputCount, linearD.OutputCount, linearD.ForwardKernel, CPU.LinearD.GetBiasedValue, linearD);
                        linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Weight, linearD.Bias.Grad, linearD.NoBias, linearD.InputCount, linearD.OutputCount, linearD.BackwardgWKernel, linearD.BackwardgXKernel, CPU.LinearD.CalcBiasGrad, linearD.Activation);
                        break;
                }
            }
        }

        public override CPU.Convolution2D<T> AsConvolution2D()
        {
            return new Convolution2D<T>(this);
        }
    }
#endif

#if DOUBLE
    public static class LinearD
#else
    public static class LinearF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real[] weight, Real[] bias, bool noBias, int inputCount, int outputCount, ComputeKernel forwardKernel, Func<int, int, Real[], Real[]> getBiasedValue, IFunction<Real> linear)
        {
            Real[] y = noBias ? new Real[outputCount * x.BatchCount] : getBiasedValue(x.BatchCount, outputCount, bias);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, weight))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, y))
            {
                forwardKernel.SetMemoryArgument(0, gpuX);
                forwardKernel.SetMemoryArgument(1, gpuW);
                forwardKernel.SetMemoryArgument(2, gpuY);
                forwardKernel.SetValueArgument(3, outputCount);
                forwardKernel.SetValueArgument(4, inputCount);

                OpenCL.CommandQueue.Execute
                    (
                        forwardKernel,
                        null,
                        new long[] { outputCount, x.BatchCount },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, new[] { outputCount }, x.BatchCount, linear);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, Real[] biasGrad, bool noBias, int inputCount, int outputCount, ComputeKernel backwardgWKernel, ComputeKernel backwardgXKernel, Action<Real[], int, int, Real[]> calcBiasGrad, KelpNet.ICompressibleActivation<Real> activation)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y) : y.Grad;
            if (!noBias)
            {
                calcBiasGrad(activatedgy, y.BatchCount, outputCount, biasGrad);
            }

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
                    backwardgWKernel.SetValueArgument(5, inputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        backwardgWKernel,
                        null,
                        new long[] { inputCount, outputCount },
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
                    backwardgXKernel.SetValueArgument(3, y.BatchCount);
                    backwardgXKernel.SetValueArgument(4, outputCount);
                    backwardgXKernel.SetValueArgument(5, inputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        backwardgXKernel,
                        null,
                        new long[] { inputCount, y.BatchCount },
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
