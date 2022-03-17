using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using System.Collections.Generic;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using KelpNet.CL.Properties;
using Real = System.Single;
using Math = System.MathF;
#else
using KelpNet.CL.Properties;
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "MaxPooling2D", Namespace = "KelpNet")]
    public class MaxPooling2D<T> : CPU.MaxPooling2D<T>, IParallelizable where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "MaxPooling2D";
        public string KernelSource => OpenCL.GetKernelSource(Resources.MaxPooling2D);

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [DataMember]
        public bool IsParallel { get; set; }

        public MaxPooling2D(int ksize, int stride = 1, int pad = 0, bool coverAll = true, bool gpuEnable = false, string name = "MaxPooling2D", string[] inputNames = null, string[] outputNames = null) : base(ksize, stride, pad, coverAll, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            InitFunc(new StreamingContext());
        }

        public MaxPooling2D(int[] ksize, int[] stride = null, int[] pad = null, bool coverAll = true, bool gpuEnable = false, string name = "MaxPooling2D", string[] inputNames = null, string[] outputNames = null) : base(ksize, stride, pad, coverAll, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            InitFunc(new StreamingContext());
        }

        public MaxPooling2D(CPU.MaxPooling2D<T> maxPooling2D) : base(maxPooling2D.Name, maxPooling2D.InputNames, maxPooling2D.OutputNames)
        {
            this.KernelHeight = maxPooling2D.KernelHeight;
            this.KernelWidth = maxPooling2D.KernelWidth;
            this.PadY = maxPooling2D.PadY;
            this.PadX = maxPooling2D.PadX;
            this.StrideX = maxPooling2D.StrideX;
            this.StrideY = maxPooling2D.StrideY;
            this.CoverAll = maxPooling2D.CoverAll;

            this.SetParallel(true);
        }

        public bool SetParallel(bool enable)
        {
            OutputIndicesList = new List<int[]>();

            this.IsParallel = enable & OpenCL.Enable;

            if (IsParallel)
            {
                ForwardKernel = OpenCL.CreateProgram<T>(KernelSource).CreateKernel("MaxPoolingForward");
            }

            InitFunc(new StreamingContext());
            return IsParallel;
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case MaxPooling2D<float> maxPooling2DF:
                        maxPooling2DF.SingleInputForward = x => MaxPooling2DF.SingleInputForward(x, maxPooling2DF.KernelWidth, maxPooling2DF.KernelHeight, maxPooling2DF.StrideX, maxPooling2DF.StrideY, maxPooling2DF.PadX, maxPooling2DF.PadY, maxPooling2DF.CoverAll, maxPooling2DF.OutputIndicesList, CPU.MaxPooling2DF.GetForwardResult, maxPooling2DF.ForwardKernel, maxPooling2DF);
                        break;

                    case MaxPooling2D<double> maxPooling2DD:
                        maxPooling2DD.SingleInputForward = x => MaxPooling2DD.SingleInputForward(x, maxPooling2DD.KernelWidth, maxPooling2DD.KernelHeight, maxPooling2DD.StrideX, maxPooling2DD.StrideY, maxPooling2DD.PadX, maxPooling2DD.PadY, maxPooling2DD.CoverAll, maxPooling2DD.OutputIndicesList, CPU.MaxPooling2DD.GetForwardResult, maxPooling2DD.ForwardKernel, maxPooling2DD);
                        break;
                }
            }
        }
    }
#endif

#if DOUBLE
    public static class MaxPooling2DD
#else
    public static class MaxPooling2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, bool coverAll, List<int[]> outputIndicesList, Func<NdArray<Real>, int[], int, int, List<int[]>, IFunction<Real>, NdArray<Real>> getForwardResult, ComputeKernel forwardKernel, IFunction<Real> maxPooling2d)
        {
            int outputHeight = coverAll ?
                (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0f + strideY - 1.0f) / strideY) + 1 :
                (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = coverAll ?
                (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0f + strideX - 1.0f) / strideX) + 1 :
                (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0f) / strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<int> gpuYIndex = new ComputeBuffer<int>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, outputIndices.Length))
            {
                forwardKernel.SetMemoryArgument(0, gpuX);
                forwardKernel.SetMemoryArgument(1, gpuYIndex);
                forwardKernel.SetValueArgument(2, outputHeight);
                forwardKernel.SetValueArgument(3, outputWidth);
                forwardKernel.SetValueArgument(4, input.Shape[0]);
                forwardKernel.SetValueArgument(5, input.Shape[1]);
                forwardKernel.SetValueArgument(6, input.Shape[2]);
                forwardKernel.SetValueArgument(7, kernelHeight);
                forwardKernel.SetValueArgument(8, kernelWidth);
                forwardKernel.SetValueArgument(9, strideX);
                forwardKernel.SetValueArgument(10, strideY);
                forwardKernel.SetValueArgument(11, padY);
                forwardKernel.SetValueArgument(12, padX);

                OpenCL.CommandQueue.Execute
                (
                    forwardKernel,
                    null,
                    new long[] { input.BatchCount * input.Shape[0], outputHeight, outputWidth },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuYIndex, ref outputIndices, true, null);
            }

            return getForwardResult(input, outputIndices, outputWidth, outputHeight, outputIndicesList, maxPooling2d);
        }
    }
}
