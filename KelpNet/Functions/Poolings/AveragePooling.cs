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

        public AveragePooling(int ksize, int stride = 1, int pad = 0, string name = "AvgPooling") : base(name)
        {
            this._kSize = ksize;
            this._stride = stride;
            this._pad = pad;
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;
            NdArray result = NdArray.Zeros(input.Shape[0], outputSize, outputSize);

            double m = this._kSize * this._kSize;

            for (int j = 0; j < input.Shape[0]; j++)
            {
                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        int resultIndex = result.GetIndex(j, y, x);
                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            for (int dx = 0; dx < this._kSize; dx++)
                            {
                                int inputIndexY = y * this._stride + dy - this._pad;
                                int inputIndexX = x * this._stride + dx - this._pad;

                                if (inputIndexY >= 0 && inputIndexY < input.Shape[1] &&
                                    inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                {
                                    result.Data[resultIndex] += input.Get(j, inputIndexY, inputIndexX) / m;
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.ZerosLike(prevInput);
            gy.Shape = (int[])prevOutput.Shape.Clone();

            double m = this._kSize * this._kSize;

            for (int j = 0; j < result.Shape[0]; j++)
            {
                for (int y = 0; y < gy.Shape[1]; y++)
                {
                    for (int x = 0; x < gy.Shape[2]; x++)
                    {
                        int gyIndex = result.GetIndex(j, y, x);

                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            for (int dx = 0; dx < this._kSize; dx++)
                            {
                                int outputIndexY = y * this._stride + dy - this._pad;
                                int outputIndexX = x * this._stride + dx - this._pad;

                                if (outputIndexY >= 0 && outputIndexY < result.Shape[1] &&
                                    outputIndexX >= 0 && outputIndexX < result.Shape[2])
                                {
                                    result.Data[result.GetIndex(j, outputIndexY, outputIndexX)] = gy.Data[gyIndex] / m;
                                }
                            }
                        }
                    }
                }
            }


            return result;
        }
    }
}
