using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "MaxPooling2D", Namespace = "KelpNet")]
    public class MaxPooling2D : CPU.MaxPooling2D, IParallelizable
    {
        public string FunctionName => "MaxPooling2D";
        public string KernelSource => OpenCL.GetKernelSource(Resources.MaxPooling2D);

        //[NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [DataMember]
        public bool IsParallel { get; set; }

        public MaxPooling2D(int ksize, int stride = 1, int pad = 0, bool coverAll = true, bool gpuEnable = false, string name = "MaxPooling2D", string[] inputNames = null, string[] outputNames = null) : base(ksize, stride, pad, coverAll, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public MaxPooling2D(int[] ksize, int[] stride = null, int[] pad = null, bool coverAll = true, bool gpuEnable = false, string name = "MaxPooling2D", string[] inputNames = null, string[] outputNames = null) : base(ksize, stride, pad, coverAll, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public MaxPooling2D(CPU.MaxPooling2D maxPooling2D) : base(maxPooling2D.Name, maxPooling2D.InputNames, maxPooling2D.OutputNames)
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
            this.IsParallel = enable & OpenCL.Enable;

            if (IsParallel)
            {
                ForwardKernel = OpenCL.CreateProgram(KernelSource).CreateKernel("MaxPoolingForward");
            }

            return IsParallel;
        }

        public override NdArray SingleInputForward(NdArray input)
        {
            //フラグチェック
            if (!IsParallel) return base.SingleInputForward(input);

            int outputHeight = CoverAll ?
                (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0 + this.StrideY - 1.0) / this.StrideY) + 1 :
                (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = CoverAll ?
                (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0 + this.StrideX - 1.0) / this.StrideX) + 1 :
                (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<int> gpuYIndex = new ComputeBuffer<int>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, outputIndices.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuYIndex);
                ForwardKernel.SetValueArgument(2, outputHeight);
                ForwardKernel.SetValueArgument(3, outputWidth);
                ForwardKernel.SetValueArgument(4, input.Shape[0]);
                ForwardKernel.SetValueArgument(5, input.Shape[1]);
                ForwardKernel.SetValueArgument(6, input.Shape[2]);
                ForwardKernel.SetValueArgument(7, this.KernelHeight);
                ForwardKernel.SetValueArgument(8, this.KernelWidth);
                ForwardKernel.SetValueArgument(9, this.StrideX);
                ForwardKernel.SetValueArgument(10, this.StrideY);
                ForwardKernel.SetValueArgument(11, this.PadY);
                ForwardKernel.SetValueArgument(12, this.PadX);

                OpenCL.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * input.Shape[0], outputHeight, outputWidth },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuYIndex, ref outputIndices, true, null);
            }

            return this.GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }
    }
}
