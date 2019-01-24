using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class MaxPooling<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "MaxPooling";

        private int _kWidth;
        private int _kHeight;
        private int _padX;
        private int _padY;
        private int _strideX;
        private int _strideY;

        private readonly List<int[]> _outputIndicesList = new List<int[]>();

        public MaxPooling(int ksize, int stride = 1, int pad = 0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._kHeight = ksize;
            this._kWidth = ksize;
            this._padY = pad;
            this._padX = pad;
            this._strideX = stride;
            this._strideY = stride;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public MaxPooling(int kWidth, int kHeight, int strideX = 1, int strideY = 1, int padX = 0, int padY = 0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._kHeight = kHeight;
            this._kWidth = kWidth;
            this._padY = padY;
            this._padX = padX;
            this._strideX = strideX;
            this._strideY = strideY;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray<T> ForwardCpu(NdArray<T> input)
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
                            Real<T> maxVal = input.Data[outputIndices[resultIndex]];

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

        NdArray<T> GetForwardResult(NdArray<T> input, int[] outputIndices, int outputWidth, int outputHeight)
        {
            Real<T>[] result = new Real<T>[outputIndices.Length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = input.Data[outputIndices[i]];
            }

            this._outputIndicesList.Add(outputIndices);

            return NdArray<T>.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        private void BackwardCpu(NdArray<T> y, NdArray<T> x)
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
