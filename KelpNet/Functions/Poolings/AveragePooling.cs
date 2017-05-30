using System;
using System.Drawing;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class AveragePooling : NeedPreviousDataFunction
    {
        private int _kHeight;
        private int _kWidth;
        private int _padY;
        private int _padX;
        private int _stride;

        public AveragePooling(int ksize, int stride = 1, int pad = 0, string name = "AvgPooling", bool isGpu = true) : base(name, isGpu)
        {
            this._kWidth = ksize;
            this._kHeight = ksize;
            this._padY = pad;
            this._padX = pad;
            this._stride = stride;
        }

        public AveragePooling(Size ksize, int stride = 1, Size pad = new Size(), string name = "AvgPooling", bool isGpu = true) : base(name, isGpu)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kWidth = ksize.Width;
            this._kHeight = ksize.Height;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._stride = stride;
        }

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._stride) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._stride) + 1;
            Real[] result = new Real[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];
            Real m = this._kHeight * this._kWidth;

            for (int b = 0; b < input.BatchCount; b++)
            {
                int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                for (int i = 0; i < input.Shape[0]; i++)
                {
                    int inputIndexOffset = i * input.Shape[1] * input.Shape[2];

                    for (int y = 0; y < outputHeight; y++)
                    {
                        int dyOffset = y * this._stride - this._padY < 0 ? 0 : y * this._stride - this._padY;
                        int dyLimit = this._kHeight + dyOffset < input.Shape[1] ? this._kHeight + dyOffset : input.Shape[1];

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int dxOffset = x * this._stride - this._padX < 0 ? 0 : x * this._stride - this._padX;
                            int dxLimit = this._kWidth + dxOffset < input.Shape[2] ? this._kWidth + dxOffset : input.Shape[2];

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int inputindex = inputIndexOffset + dy * input.Shape[2] + dx;
                                    result[resultIndex] += input.Data[inputindex + input.Length * b] / m;
                                }
                            }

                            resultIndex++;
                        }
                    }
                }
            }

            return BatchArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput, BatchArray prevOutput)
        {
            Real[] result = new Real[prevInput.Data.Length];
            Real m = this._kHeight * this._kWidth;

            for (int b = 0; b < gy.BatchCount; b++)
            {
                int gyIndex = b * gy.Length;

                for (int i = 0; i < prevInput.Shape[0]; i++)
                {
                    int resultIndexOffset = b * prevInput.Length + i * prevInput.Shape[1] * prevInput.Shape[2];

                    for (int y = 0; y < prevOutput.Shape[1]; y++)
                    {
                        int dyOffset = y * this._stride - this._padY < 0 ? 0 : y * this._stride - this._padY;
                        int dyLimit = this._kHeight + dyOffset < prevInput.Shape[1] ? this._kHeight + dyOffset : prevInput.Shape[1];

                        for (int x = 0; x < prevOutput.Shape[2]; x++)
                        {
                            int dxOffset = x * this._stride - this._padX < 0 ? 0 : x * this._stride - this._padX;
                            int dxLimit = this._kWidth + dxOffset < prevInput.Shape[2] ? this._kWidth + dxOffset : prevInput.Shape[2];

                            Real gyData = gy.Data[gyIndex] / m;

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int resultIndex = resultIndexOffset + dy * prevInput.Shape[2] + dx;
                                    result[resultIndex] = gyData;
                                }
                            }

                            gyIndex++;
                        }
                    }
                }
            }

            return BatchArray.Convert(result, prevInput.Shape, gy.BatchCount);
        }
    }
}
