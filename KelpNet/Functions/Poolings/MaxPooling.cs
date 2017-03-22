using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class MaxPooling : Function
    {
        private int _kWidth;
        private int _kHeight;
        private int _padX;
        private int _padY;
        private int _stride;

        private readonly List<int[]> _outputIndicesList = new List<int[]>();
        private int[] _prevInputShape;
        private int _prevInputDataLength;
        private int _prevInputBatchCount;

        public MaxPooling(int ksize, int stride = 1, int pad = 0, bool isGpu = true, string name = "MaxPooling") : base(name)
        {
            this._kHeight = ksize;
            this._kWidth = ksize;
            this._padY = pad;
            this._padX = pad;
            this._stride = stride;

            //カーネルを作成
            if (isGpu)
            {
                initGPU();
            }
        }

        public MaxPooling(Size ksize, int stride = 1, Size pad = new Size(), bool isGpu = true, string name = "MaxPooling") : base(name)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kHeight = ksize.Height;
            this._kWidth = ksize.Width;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._stride = stride;

            //カーネルを作成
            if (isGpu)
            {
                initGPU();
            }
        }

        void initGPU()
        {
            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "MaxPoolingForward");

            //メモリ転送のみの為不要
            //BackwardKernel = Weaver.CreateKernel("", "");
        }

        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void MaxPoolingForward(
	__global const double *gpuX,
	__global int *gpuYindex,
    const int outputHeight, const int outputWidth,
    const int inputShape0, const int inputShape1, const int inputShape2,    
    const int kHeight, const int kWidth,
    const int stride,
    const int padY, const int padX)
{
	int b = get_global_id(0) / inputShape0;
	int i = get_global_id(0) % inputShape0;
    int y = get_global_id(1);
    int x = get_global_id(2);

    int resultIndex = b * inputShape0 * outputHeight * outputWidth + i * outputHeight * outputWidth + y * outputWidth + x;
    int inputLength = inputShape0 * inputShape1 * inputShape2;

    int inputIndexOffset = i * inputShape1 * inputShape2;

    double maxVal = -DBL_MAX;

    for (int dy = 0; dy < kHeight; dy++)
    {
        int inputIndexY = y * stride + dy - padY;

        if (inputIndexY >= 0 && inputIndexY < inputShape1)
        {
            for (int dx = 0; dx < kWidth; dx++)
            {
                int inputIndexX = x * stride + dx - padX;

                if (inputIndexX >= 0 && inputIndexX < inputShape2)
                {
                    int inputIndex = inputIndexOffset + inputIndexY * inputShape2 + inputIndexX + b * inputLength;

                    if (maxVal < gpuX[inputIndex])
                    {
                        maxVal = gpuX[inputIndex];
                        gpuYindex[resultIndex] = inputIndex;
                    }
                }
            }
        }
    }
}";
        protected override BatchArray ForwardSingle(BatchArray input, bool isGpu)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._stride) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._stride) + 1;
            double[] result = Enumerable.Repeat(double.MinValue, input.Shape[0] * outputHeight * outputWidth * input.BatchCount).ToArray();
            int[] outputIndices = new int[result.Length];
            this._prevInputShape = input.Shape.ToArray();
            this._prevInputDataLength = input.Data.Length;
            this._prevInputBatchCount = input.BatchCount;

            if (!isGpu)
            {
                for (int b = 0; b < input.BatchCount; b++)
                {
                    int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                    for (int i = 0; i < input.Shape[0]; i++)
                    {
                        int inputIndexOffset = i * input.Shape[1] * input.Shape[2];

                        for (int y = 0; y < outputHeight; y++)
                        {
                            for (int x = 0; x < outputWidth; x++)
                            {
                                double maxVal = double.MinValue;
                                for (int dy = 0; dy < this._kHeight; dy++)
                                {
                                    int inputIndexY = y * this._stride + dy - this._padY;

                                    if (inputIndexY >= 0 && inputIndexY < input.Shape[1])
                                    {
                                        for (int dx = 0; dx < this._kWidth; dx++)
                                        {
                                            int inputIndexX = x * this._stride + dx - this._padX;

                                            if (inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                            {
                                                int inputIndex = inputIndexOffset + inputIndexY * input.Shape[2] + inputIndexX + b * input.Length;
                                                if (maxVal < input.Data[inputIndex])
                                                {
                                                    maxVal = input.Data[inputIndex];
                                                    outputIndices[resultIndex] = inputIndex;
                                                }
                                            }
                                        }
                                    }
                                }

                                resultIndex++;
                            }
                        }
                    }
                }
            }
            else
            {
                using (ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
                using (ComputeBuffer<int> gpuYIndex = new ComputeBuffer<int>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, outputIndices))
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
                    ForwardKernel.SetValueArgument(9, this._stride);
                    ForwardKernel.SetValueArgument(10, this._padY);
                    ForwardKernel.SetValueArgument(11, this._padX);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
                            null,
                            new long[] {input.BatchCount* input.Shape[0],  outputHeight , outputWidth },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuYIndex, ref outputIndices, true, null);
                }
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = input.Data[outputIndices[i]];
            }
            this._outputIndicesList.Add(outputIndices);

            return BatchArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount);
        }

        protected override BatchArray BackwardSingle(BatchArray gy, bool isGpu)
        {
            int[] outputIndices = this._outputIndicesList[this._outputIndicesList.Count - 1];
            this._outputIndicesList.RemoveAt(this._outputIndicesList.Count - 1);

            double[] result = new double[this._prevInputDataLength];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                result[outputIndices[i]] = gy.Data[i];
            }

            return BatchArray.Convert(result, this._prevInputShape, this._prevInputBatchCount);
        }
    }
}
