using System;
using System.Drawing;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class AveragePooling : SingleInputFunction
    {
        private int _kHeight;
        private int _kWidth;
        private int _padY;
        private int _padX;
        private int _strideX;
        private int _strideY;

        public AveragePooling(int ksize, int stride = 1, int pad = 0, string name = "AvgPooling") : base(name)
        {
            this._kWidth = ksize;
            this._kHeight = ksize;
            this._padY = pad;
            this._padX = pad;
            this._strideX = stride;
            this._strideY = stride;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        public AveragePooling(Size ksize, Size stride = new Size(), Size pad = new Size(), string name = "AvgPooling") : base(name)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(0, 0);

            this._kWidth = ksize.Width;
            this._kHeight = ksize.Height;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._strideX = stride.Width;
            this._strideY = stride.Height;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;
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
                        int dyOffset = y * this._strideY - this._padY < 0 ? 0 : y * this._strideY - this._padY;
                        int dyLimit = this._kHeight + dyOffset < input.Shape[1] ? this._kHeight + dyOffset : input.Shape[1];

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int dxOffset = x * this._strideX - this._padX < 0 ? 0 : x * this._strideX - this._padX;
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

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real m = this._kHeight * this._kWidth;

            for (int b = 0; b < y.BatchCount; b++)
            {
                int gyIndex = b * y.Length;

                for (int i = 0; i < x.Shape[0]; i++)
                {
                    int resultIndexOffset = b * x.Length + i * x.Shape[1] * x.Shape[2];

                    for (int posY = 0; posY < y.Shape[1]; posY++)
                    {
                        int dyOffset = posY * this._strideY - this._padY < 0 ? 0 : posY * this._strideY - this._padY;
                        int dyLimit = this._kHeight + dyOffset < x.Shape[1] ? this._kHeight + dyOffset : x.Shape[1];

                        for (int posX = 0; posX < y.Shape[2]; posX++)
                        {
                            int dxOffset = posX * this._strideX - this._padX < 0 ? 0 : posX * this._strideX - this._padX;
                            int dxLimit = this._kWidth + dxOffset < x.Shape[2] ? this._kWidth + dxOffset : x.Shape[2];

                            Real gyData = y.Grad[gyIndex] / m;

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int resultIndex = resultIndexOffset + dy * x.Shape[2] + dx;
                                    x.Grad[resultIndex] += gyData;
                                }
                            }

                            gyIndex++;
                        }
                    }
                }
            }
        }
    }
}
