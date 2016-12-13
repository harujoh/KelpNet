using System;
using KelpNet.Common;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class AveragePooling : NeedPreviousDataFunction
    {
        private int _kSize;
        private int _stride;
        private int _pad;

        public AveragePooling(int ksize, int stride = 1, int pad = 0, string name = "AvgPooling", bool isParallel = true) : base(name, isParallel)
        {
            this._kSize = ksize;
            this._stride = stride;
            this._pad = pad;
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;
            double[] result = new double[input.Shape[0] * outputSize * outputSize];

            double m = this._kSize * this._kSize;

            int resultIndex = 0;

            for (int i = 0; i < input.Shape[0]; i++)
            {
                int inputIndexOffset = i * input.Shape[1] * input.Shape[2];

                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            int inputIndexY = y * this._stride + dy - this._pad;

                            if (inputIndexY >= 0 && inputIndexY < input.Shape[1])
                            {
                                for (int dx = 0; dx < this._kSize; dx++)
                                {
                                    int inputIndexX = x * this._stride + dx - this._pad;

                                    if (inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                    {
                                        int inputindex = inputIndexOffset + inputIndexY * input.Shape[2] + inputIndexX;

                                        result[resultIndex] += input.Data[inputindex] / m;
                                    }
                                }
                            }
                        }

                        resultIndex++;
                    }
                }
            }

            return NdArray.Convert(result, new[] { input.Shape[0], outputSize, outputSize });
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] result = new double[prevInput.Length];

            double m = this._kSize * this._kSize;

            int gyIndex = 0;

            for (int i = 0; i < prevInput.Shape[0]; i++)
            {
                int resultIndexOffset = i * prevInput.Shape[1] * prevInput.Shape[2];
                for (int y = 0; y < prevOutput.Shape[1]; y++)
                {
                    for (int x = 0; x < prevOutput.Shape[2]; x++)
                    {
                        double gyData = gy.Data[gyIndex] / m;

                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            int outputIndexY = y * this._stride + dy - this._pad;

                            if (outputIndexY >= 0 && outputIndexY < prevInput.Shape[1])
                            {
                                for (int dx = 0; dx < this._kSize; dx++)
                                {
                                    int outputIndexX = x * this._stride + dx - this._pad;

                                    if (outputIndexX >= 0 && outputIndexX < prevInput.Shape[2])
                                    {
                                        int resultIndex = resultIndexOffset + outputIndexY * prevInput.Shape[2] + outputIndexX;
                                        result[resultIndex] = gyData;
                                    }
                                }
                            }
                        }

                        gyIndex++;
                    }
                }
            }

            return NdArray.Convert(result, prevInput.Shape);
        }
    }
}
