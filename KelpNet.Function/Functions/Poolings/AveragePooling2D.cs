using System;
using System.Runtime.Serialization;
using KelpNet.CPU;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class AveragePooling2D<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
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

            InitFunc(new StreamingContext());
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

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case AveragePooling2D<float> averagePooling2DF:
                    averagePooling2DF.SingleInputForward = x => AveragePooling2DF.SingleInputForward(x, averagePooling2DF.KernelWidth, averagePooling2DF.KernelHeight, averagePooling2DF.StrideX, averagePooling2DF.StrideY, averagePooling2DF.PadX, averagePooling2DF.PadY, averagePooling2DF);
                    averagePooling2DF.SingleOutputBackward = (y, x) => AveragePooling2DF.SingleOutputBackward(y, x, averagePooling2DF.KernelWidth, averagePooling2DF.KernelHeight, averagePooling2DF.StrideX, averagePooling2DF.StrideY, averagePooling2DF.PadX, averagePooling2DF.PadY);
                    break;

                case AveragePooling2D<double> averagePooling2DD:
                    averagePooling2DD.SingleInputForward = x => AveragePooling2DD.SingleInputForward(x, averagePooling2DD.KernelWidth, averagePooling2DD.KernelHeight, averagePooling2DD.StrideX, averagePooling2DD.StrideY, averagePooling2DD.PadX, averagePooling2DD.PadY, averagePooling2DD);
                    averagePooling2DD.SingleOutputBackward = (y, x) => AveragePooling2DD.SingleOutputBackward(y, x, averagePooling2DD.KernelWidth, averagePooling2DD.KernelHeight, averagePooling2DD.StrideX, averagePooling2DD.StrideY, averagePooling2DD.PadX, averagePooling2DD.PadY);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AveragePooling2DD
#else
    public static class AveragePooling2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, IFunction<Real> avgPooling2d)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0) / strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0) / strideX) + 1;
            Real[] result = new Real[input.BatchCount * input.Shape[0] * outputHeight * outputWidth];
            Real m = kernelHeight * kernelWidth;

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
                        int inIndexY = y * strideY - padY;
                        int dyLimit = kernelHeight < input.Shape[1] - inIndexY ? kernelHeight : input.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int inIndexX = x * strideX - padX;
                            int dxLimit = kernelWidth < input.Shape[2] - inIndexX ? kernelWidth : input.Shape[2] - inIndexX;
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

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, avgPooling2d);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY)
        {
            Real m = kernelHeight * kernelWidth;

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
                        int inIndexY = outY * strideY - padY;
                        int dyLimit = kernelHeight < x.Shape[1] - inIndexY ? kernelHeight : x.Shape[1] - inIndexY;
                        int dyStart = inIndexY < 0 ? -inIndexY : 0;

                        for (int outX = 0; outX < y.Shape[2]; outX++)
                        {
                            int inIndexX = outX * strideX - padX;
                            int dxLimit = kernelWidth < x.Shape[2] - inIndexX ? kernelWidth : x.Shape[2] - inIndexX;
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
