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

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        public MaxPooling(int ksize, int stride = 1, int pad = 0, string name = "MaxPooling", bool isGpu = true) : base(name, isGpu)
        {
            this._kHeight = ksize;
            this._kWidth = ksize;
            this._padY = pad;
            this._padX = pad;
            this._stride = stride;

            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "MaxPoolingForward");
            }
        }

        public MaxPooling(Size ksize, int stride = 1, Size pad = new Size(), string name = "MaxPooling", bool isGpu = true) : base(name, isGpu)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kHeight = ksize.Height;
            this._kWidth = ksize.Width;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._stride = stride;

            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "MaxPoolingForward");
            }
        }


        public string ForwardKernelSource { get; } =
@"
__kernel void MaxPoolingForward(
	__global const Real *gpuX,
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

    int indexOffset = b * inputShape0 * inputShape1 * inputShape2 + i * inputShape1 * inputShape2;

    int dyOffset = y * stride - padY < 0 ? 0 : y * stride - padY;
    int dyLimit = kHeight + dyOffset < inputShape1 ? kHeight + dyOffset : inputShape1;

    int dxOffset = x * stride - padX < 0 ? 0 : x * stride - padX;
    int dxLimit = kWidth + dxOffset < inputShape2 ? kWidth + dxOffset : inputShape2;

    int yIndex = indexOffset + dyOffset * inputShape2 + dxOffset;
    Real maxVal = gpuX[yIndex];

    for (int dy = dyOffset; dy < dyLimit; dy++)
    {
        for (int dx = dxOffset; dx < dxLimit; dx++)
        {
            int inputIndex = indexOffset + dy * inputShape2 + dx;

            if (maxVal < gpuX[inputIndex])
            {
                maxVal = gpuX[inputIndex];
                yIndex = inputIndex;
            }
        }
    }

    gpuYindex[b * inputShape0 * outputHeight * outputWidth + i * outputHeight * outputWidth + y * outputWidth + x] = yIndex;
}";

        protected override BatchArray ForwardSingle(BatchArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._stride) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._stride) + 1;
            Real[] result = new Real[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];
            int[] outputIndices = new int[result.Length];
            this._prevInputShape = input.Shape.ToArray();
            this._prevInputDataLength = input.Data.Length;
            this._prevInputBatchCount = input.BatchCount;

            if (!IsGpu)
            {
                for (int b = 0; b < input.BatchCount; b++)
                {
                    int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                    for (int i = 0; i < input.Shape[0]; i++)
                    {
                        int inputIndexOffset = b * input.Length + i * input.Shape[1] * input.Shape[2];

                        for (int y = 0; y < outputHeight; y++)
                        {
                            int dyOffset = y * this._stride + -this._padY < 0 ? 0 : y * this._stride + -this._padY;
                            int dyLimit = this._kHeight + dyOffset < input.Shape[1] ? this._kHeight + dyOffset : input.Shape[1];

                            for (int x = 0; x < outputWidth; x++)
                            {
                                int dxOffset = x * this._stride - this._padX < 0 ? 0 : x * this._stride - this._padX;
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
            }
            else
            {
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
                    ForwardKernel.SetValueArgument(9, this._stride);
                    ForwardKernel.SetValueArgument(10, this._padY);
                    ForwardKernel.SetValueArgument(11, this._padX);

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
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = input.Data[outputIndices[i]];
            }
            this._outputIndicesList.Add(outputIndices);

            return BatchArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount);
        }

        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            int[] outputIndices = this._outputIndicesList[this._outputIndicesList.Count - 1];
            this._outputIndicesList.RemoveAt(this._outputIndicesList.Count - 1);

            Real[] result = new Real[this._prevInputDataLength];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                result[outputIndices[i]] = gy.Data[i];
            }

            return BatchArray.Convert(result, this._prevInputShape, this._prevInputBatchCount);
        }
    }
}
