using System;

namespace KelpNet
{
    [Serializable]
    public class AveragePooling2D : SingleInputFunction
    {
        const string FUNCTION_NAME = "AveragePooling";

        private int _kHeight;
        private int _kWidth;
        private int _padY;
        private int _padX;
        private int _strideX;
        private int _strideY;

        public AveragePooling2D(int ksize, int stride = 1, int pad = 0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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

        public AveragePooling2D(int[] ksize, int[] stride = null, int[] pad = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this._kWidth = ksize[0];
            this._kHeight = ksize[1];
            this._padX = pad[0];
            this._padY = pad[1];
            this._strideX = stride[0];
            this._strideY = stride[1];

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;
            Real[] result = new Real[input.BatchCount * input.Shape[0] * outputHeight * outputWidth];
            Real m = this._kHeight * this._kWidth;

            for (int b = 0; b < input.BatchCount; b++)
            {
                int outBatchOffset = b * input.Shape[0] * outputHeight * outputWidth;
                int inBatchOffset = b * input.Length;

                for (int i = 0; i < input.Shape[0]; i++)
                {
                    int outChOffset = outBatchOffset + i * outputHeight * outputWidth;
                    int inChOffset = inBatchOffset + i * input.Shape[1] * input.Shape[2];

                    for (int y = 0; y < outputHeight; y++)
                    {
                        int inIndexY = y * _strideY - _padY;
                        int dyLimit = this._kHeight < input.Shape[1] - inIndexY ? this._kHeight : input.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int inIndexX = x * _strideX - _padX;
                            int dxLimit = this._kWidth < input.Shape[2] - inIndexX ? this._kWidth : input.Shape[2] - inIndexX;
                            int dxStart = inIndexX < 0 ? -inIndexX : 0;

                            int inBaseIndex = inChOffset + inIndexY * input.Shape[2] + inIndexX;
                            int outIndex = outChOffset + y * outputWidth + x;

                            for (int dy = dyStart; dy < dyLimit; dy++)
                            {
                                for (int dx = dxStart; dx < dxLimit; dx++)
                                {
                                    int inputIndex = inBaseIndex + dy * input.Shape[2] + dx;

                                    result[outIndex] += input.Data[inputIndex] / m;
                                }
                            }
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real m = this._kHeight * this._kWidth;

            for (int b = 0; b < x.BatchCount; b++)
            {
                int outBatchOffset = b * y.Shape[0] * y.Shape[1] * y.Shape[2];
                int inBatchOffset = b * x.Length;

                for (int i = 0; i < x.Shape[0]; i++)
                {
                    int outChOffset = outBatchOffset + i * y.Shape[1] * y.Shape[2];
                    int inChOffset = inBatchOffset + i * x.Shape[1] * x.Shape[2];

                    for (int outY = 0; outY < y.Shape[1]; outY++)
                    {
                        int inIndexY = outY * _strideY - _padY;
                        int dyLimit = this._kHeight < x.Shape[1] - inIndexY ? this._kHeight : x.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int outX = 0; outX < y.Shape[2]; outX++)
                        {
                            int inIndexX = outX * _strideX - _padX;
                            int dxLimit = this._kWidth < x.Shape[2] - inIndexX ? this._kWidth : x.Shape[2] - inIndexX;
                            int dxStart = inIndexX < 0 ? -inIndexX : 0;

                            int inBaseIndex = inChOffset + inIndexY * x.Shape[2] + inIndexX;
                            int outIndex = outChOffset + outY * y.Shape[2] + outX;

                            Real gyData = y.Grad[outIndex] / m;

                            for (int dy = dyStart; dy < dyLimit; dy++)
                            {
                                for (int dx = dxStart; dx < dxLimit; dx++)
                                {
                                    int inputIndex = inBaseIndex + dy * x.Shape[2] + dx;

                                    x.Grad[inputIndex] += gyData;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
