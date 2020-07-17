using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "MaxPooling2D", Namespace = "KelpNet")]
    public class MaxPooling2D<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "MaxPooling2D";

        [DataMember]
        public int KernelWidth;

        [DataMember]
        public int KernelHeight;

        [DataMember]
        public int PadX;

        [DataMember]
        public int PadY;

        [DataMember]
        public int StrideX;

        [DataMember]
        public int StrideY;

        [DataMember]
        public bool CoverAll;

        protected List<int[]> OutputIndicesList = new List<int[]>();

        public MaxPooling2D(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public MaxPooling2D(int kernelSize, int stride = 1, int pad = 0, bool coverAll = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.KernelHeight = kernelSize;
            this.KernelWidth = kernelSize;
            this.PadY = pad;
            this.PadX = pad;
            this.StrideX = stride;
            this.StrideY = stride;
            this.CoverAll = coverAll;

            InitFunc(new StreamingContext());
        }

        public MaxPooling2D(int[] kernelSize, int[] stride = null, int[] pad = null, bool coverAll = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
            this.CoverAll = coverAll;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case MaxPooling2D<float> maxPooling2DF:
                    maxPooling2DF.SingleInputForward = x => MaxPooling2DF.SingleInputForward(x, maxPooling2DF.KernelWidth, maxPooling2DF.KernelHeight, maxPooling2DF.StrideX, maxPooling2DF.StrideY, maxPooling2DF.PadX, maxPooling2DF.PadY, maxPooling2DF.CoverAll, maxPooling2DF.OutputIndicesList, maxPooling2DF);
                    maxPooling2DF.SingleOutputBackward = (y, x) => MaxPooling2DF.SingleOutputBackward(y, x, maxPooling2DF.OutputIndicesList);
                    break;

                case MaxPooling2D<double> maxPooling2DD:
                    maxPooling2DD.SingleInputForward = x => MaxPooling2DD.SingleInputForward(x, maxPooling2DD.KernelWidth, maxPooling2DD.KernelHeight, maxPooling2DD.StrideX, maxPooling2DD.StrideY, maxPooling2DD.PadX, maxPooling2DD.PadY, maxPooling2DD.CoverAll, maxPooling2DD.OutputIndicesList, maxPooling2DD);
                    maxPooling2DD.SingleOutputBackward = (y, x) => MaxPooling2DD.SingleOutputBackward(y, x, maxPooling2DD.OutputIndicesList);
                    break;
            }
        }

        public override void ResetState()
        {
            base.ResetState();
            this.OutputIndicesList = new List<int[]>();
        }
    }
#endif

#if DOUBLE
    public static class MaxPooling2DD
#else
    public static class MaxPooling2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, bool coverAll, List<int[]> outputIndicesList, IFunction<Real> maxPooling2d)
        {
            int outputHeight = coverAll ?
                (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0f + strideY - 1.0f) / strideY) + 1 :
                (int)Math.Floor((input.Shape[1] - kernelHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = coverAll ?
                (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0f + strideX - 1.0f) / strideX) + 1 :
                (int)Math.Floor((input.Shape[2] - kernelWidth + padX * 2.0f) / strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            for (int i = 0; i < outputIndices.Length; i++)
            {
                outputIndices[i] = -1;
            }

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

                            Real maxVal = float.NegativeInfinity;
                            outputIndices[outIndex] = -1;

                            for (int dy = dyStart; dy < dyLimit; dy++)
                            {
                                for (int dx = dxStart; dx < dxLimit; dx++)
                                {
                                    int inputIndex = inBaseIndex + dy * input.Shape[2] + dx;

                                    if (maxVal < input.Data[inputIndex])
                                    {
                                        maxVal = input.Data[inputIndex];
                                        outputIndices[outIndex] = inputIndex;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight, outputIndicesList, maxPooling2d);
        }

        public static NdArray<Real> GetForwardResult(NdArray<Real> input, int[] outputIndices, int outputWidth, int outputHeight, List<int[]> outputIndicesList, IFunction<Real> maxPooling2d)
        {
            Real[] result = new Real[outputIndices.Length];

            for (int i = 0; i < result.Length; i++)
            {
                if (outputIndices[i] == -1)
                {
                    result[i] = float.NegativeInfinity;
                }
                else
                {
                    result[i] = input.Data[outputIndices[i]];
                }
            }

            outputIndicesList.Add(outputIndices);

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, maxPooling2d);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, List<int[]> outputIndicesList)
        {
            int[] outputIndices = outputIndicesList[outputIndicesList.Count - 1];
            outputIndicesList.RemoveAt(outputIndicesList.Count - 1);

            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (outputIndices[i] != -1)
                {
                    x.Grad[outputIndices[i]] += y.Grad[i];
                }
            }
        }

    }
}
