using System;

namespace KelpNet.Functions.Poolings
{
    public class AveragePooling : Function, IPredictableFunction
    {
        private int _kSize;
        private int _stride;
        private int _pad;

        public AveragePooling(int ksize, int stride = 1, int pad = 0)
        {
            this._kSize = ksize;
            this._stride = stride;
            this._pad = pad;
        }

        public override NdArray Forward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;
            NdArray result = NdArray.Zeros(input.Shape[0], outputSize, outputSize);

            double m = this._kSize * this._kSize;

            for (int i = 0; i < input.Shape[0]; i++)
            {
                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            for (int dx = 0; dx < this._kSize; dx++)
                            {
                                int inputIndexY = y * this._stride + dy - this._pad;
                                int inputIndexX = x * this._stride + dx - this._pad;

                                if (inputIndexY >= 0 && inputIndexY < input.Shape[1] &&
                                    inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                {
                                    result.Data[result.GetIndex(i, y, x)] += input.Get(i, inputIndexY, inputIndexX) / m;
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        public override NdArray Backward(NdArray gy, NdArray PrevInput, NdArray PrevOutput)
        {
            NdArray result = NdArray.EmptyLike(PrevInput);
            gy.Shape = PrevOutput.Shape;
            double m = this._kSize * this._kSize;

            for (int i = 0; i < result.Shape[0]; i++)
            {
                for (int y = 0; y < gy.Shape[1]; y++)
                {
                    for (int x = 0; x < gy.Shape[2]; x++)
                    {
                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            for (int dx = 0; dx < this._kSize; dx++)
                            {
                                int outputIndexY = y * this._stride + dy - this._pad;
                                int outputIndexX = x * this._stride + dx - this._pad;

                                if (outputIndexY >= 0 && outputIndexY < result.Shape[1] &&
                                    outputIndexX >= 0 && outputIndexX < result.Shape[2])
                                {
                                    var index = gy.GetIndex(i, y, x);
                                    result.Data[result.GetIndex(i, outputIndexY, outputIndexX)] = gy.Data[index] / m;
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
