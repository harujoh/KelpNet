using System;

namespace KelpNet
{
    [Serializable]
    public class AveragePooling2D : SingleInputFunction
    {
        const string FUNCTION_NAME = "AveragePooling";

        public int KernelHeight;
        public int KernelWidth;
        public int PadY;
        public int PadX;
        public int StrideX;
        public int StrideY;

        public AveragePooling2D(int kernelSize, int stride = 1, int pad = 0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.KernelWidth = kernelSize;
            this.KernelHeight = kernelSize;
            this.PadY = pad;
            this.PadX = pad;
            this.StrideX = stride;
            this.StrideY = stride;
        }

        public AveragePooling2D(int[] kernelSize, int[] stride = null, int[] pad = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this.KernelWidth = kernelSize[0];
            this.KernelHeight = kernelSize[1];
            this.PadX = pad[0];
            this.PadY = pad[1];
            this.StrideX = stride[0];
            this.StrideY = stride[1];
        }

        public override NdArray SingleInputForward(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;
            Real[] result = new Real[input.BatchCount * input.Shape[0] * outputHeight * outputWidth];
            Real m = this.KernelHeight * this.KernelWidth;

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
                        int inIndexY = y * StrideY - PadY;
                        int dyLimit = this.KernelHeight < input.Shape[1] - inIndexY ? this.KernelHeight : input.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int inIndexX = x * StrideX - PadX;
                            int dxLimit = this.KernelWidth < input.Shape[2] - inIndexX ? this.KernelWidth : input.Shape[2] - inIndexX;
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

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            Real m = this.KernelHeight * this.KernelWidth;

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
                        int inIndexY = outY * StrideY - PadY;
                        int dyLimit = this.KernelHeight < x.Shape[1] - inIndexY ? this.KernelHeight : x.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int outX = 0; outX < y.Shape[2]; outX++)
                        {
                            int inIndexX = outX * StrideX - PadX;
                            int dxLimit = this.KernelWidth < x.Shape[2] - inIndexX ? this.KernelWidth : x.Shape[2] - inIndexX;
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
