using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Cloo;
using KelpNet.Properties;

namespace KelpNet
{
    [Serializable]
    public class MaxPooling2D : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "MaxPooling";

        private int _kWidth;
        private int _kHeight;
        private int _padX;
        private int _padY;
        private int _strideX;
        private int _strideY;

        private readonly List<int[]> _outputIndicesList = new List<int[]>();

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        public MaxPooling2D(int ksize, int stride = 1, int pad = 0, bool gpuEnable = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._kHeight = ksize;
            this._kWidth = ksize;
            this._padY = pad;
            this._padX = pad;
            this._strideX = stride;
            this._strideY = stride;

            this.SetGpuEnable(gpuEnable);

            SingleOutputBackward = BackwardCpu;
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            CreateKernel();

            if (GpuEnable)
            {
                SingleInputForward = ForwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
            }

            return GpuEnable;
        }

        public MaxPooling2D(Size ksize, Size stride = new Size(), Size pad = new Size(), bool gpuEnable = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(1, 1);

            this._kHeight = ksize.Height;
            this._kWidth = ksize.Width;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._strideX = stride.Width;
            this._strideY = stride.Height;

            if (this.SetGpuEnable(gpuEnable))
            {
                CreateKernel();
                SingleInputForward = ForwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
            }

            SingleOutputBackward = BackwardCpu;
        }

        public void CreateKernel()
        {
            if (GpuEnable)
                ForwardKernel = Weaver.CreateProgram(Weaver.GetKernelSource(Resources.MaxPooling)).CreateKernel("MaxPoolingForward");
        }

        private NdArray ForwardCpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            for (int b = 0; b < input.BatchCount; b++)
            {
                int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                for (int i = 0; i < input.Shape[0]; i++)
                {
                    int inputIndexOffset = b * input.Length + i * input.Shape[1] * input.Shape[2];

                    for (int y = 0; y < outputHeight; y++)
                    {
                        int dyOffset = y * this._strideY + -this._padY < 0 ? 0 : y * this._strideY + -this._padY;
                        int dyLimit = this._kHeight + dyOffset < input.Shape[1] ? this._kHeight + dyOffset : input.Shape[1];

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int dxOffset = x * this._strideX - this._padX < 0 ? 0 : x * this._strideX - this._padX;
                            int dxLimit = this._kWidth + dxOffset < input.Shape[2] ? this._kWidth + dxOffset : input.Shape[2];

                            outputIndices[resultIndex] = inputIndexOffset + dyOffset * input.Shape[2] + dxOffset;
                            Real maxVal = input.Data[outputIndices[resultIndex]];

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int inputIndex = inputIndexOffset + dy * input.Shape[2] + dx;

                                    if (maxVal < input.Data[inputIndex])
                                    {
                                        maxVal = input.Data[inputIndex];
                                        outputIndices[resultIndex] = inputIndex;
                                    }
                                }
                            }

                            resultIndex++;
                        }
                    }
                }

            }

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }

        private NdArray ForwardGpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
            using (ComputeBuffer<int> gpuYIndex = new ComputeBuffer<int>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, outputIndices.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuYIndex);
                ForwardKernel.SetValueArgument(2, outputHeight);
                ForwardKernel.SetValueArgument(3, outputWidth);
                ForwardKernel.SetValueArgument(4, input.Shape[0]);
                ForwardKernel.SetValueArgument(5, input.Shape[1]);
                ForwardKernel.SetValueArgument(6, input.Shape[2]);
                ForwardKernel.SetValueArgument(7, this._kHeight);
                ForwardKernel.SetValueArgument(8, this._kWidth);
                ForwardKernel.SetValueArgument(9, this._strideX);
                ForwardKernel.SetValueArgument(10, this._strideY);
                ForwardKernel.SetValueArgument(11, this._padY);
                ForwardKernel.SetValueArgument(12, this._padX);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * input.Shape[0], outputHeight, outputWidth },
                    null,
                    null
                );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuYIndex, ref outputIndices, true, null);
            }

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }

        NdArray GetForwardResult(NdArray input, int[] outputIndices, int outputWidth, int outputHeight)
        {
            Real[] result = new Real[outputIndices.Length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = input.Data[outputIndices[i]];
            }

            this._outputIndicesList.Add(outputIndices);

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        private void BackwardCpu(NdArray y, NdArray x)
        {
            int[] outputIndices = this._outputIndicesList[this._outputIndicesList.Count - 1];
            this._outputIndicesList.RemoveAt(this._outputIndicesList.Count - 1);

            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[outputIndices[i]] += y.Grad[i];
            }
        }
    }
}
