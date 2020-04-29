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
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real[] Weight, Real[] bias, bool noBias, int inputCount, int outputCount, ComputeKernel ForwardKernel, Func<int, int, Real[], Real[]> GetBiasedValue, IFunction<Real> linear)
        {
            Real[] y = noBias ? new Real[outputCount * x.BatchCount] : GetBiasedValue(x.BatchCount, outputCount, bias);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, Weight))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, y))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpuY);
                ForwardKernel.SetValueArgument(3, outputCount);
                ForwardKernel.SetValueArgument(4, inputCount);

                OpenCL.CommandQueue.Execute
                    (
                        ForwardKernel,
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

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, Real[] biasGrad, bool noBias, int inputCount, int outputCount, ComputeKernel BackwardgWKernel, ComputeKernel BackwardgXKernel, Action<Real[], int, int, Real[]> CalcBiasGrad, KelpNet.ICompressibleActivation<Real> Activation)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = Activation != null ? Activation.GetActivatedgy(y) : y.Grad;
            if (!noBias)
            {
                CalcBiasGrad(activatedgy, y.BatchCount, outputCount, biasGrad);
            }

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
                    BackwardgWKernel.SetValueArgument(5, inputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgWKernel,
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
                    BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    BackwardgXKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgXKernel.SetValueArgument(4, outputCount);
                    BackwardgXKernel.SetValueArgument(5, inputCount);

                    OpenCL.CommandQueue.Execute
                    (
                        BackwardgXKernel,
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
