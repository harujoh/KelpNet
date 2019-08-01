using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "MaxPooling2D", Namespace = "KelpNet")]
    public class MaxPooling2D : SingleInputFunction
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

        //[NonSerialized]
        protected List<int[]> _outputIndicesList = new List<int[]>();

        public MaxPooling2D(int kernelSize, int stride = 1, int pad = 0, bool coverAll = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.KernelHeight = kernelSize;
            this.KernelWidth = kernelSize;
            this.PadY = pad;
            this.PadX = pad;
            this.StrideX = stride;
            this.StrideY = stride;
            this.CoverAll = coverAll;
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
        }

        public MaxPooling2D(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray SingleInputForward(NdArray input)
        {
            int outputHeight = CoverAll ?
                (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0 + this.StrideY - 1.0) / this.StrideY) + 1 :
                (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = CoverAll ?
                (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0 + this.StrideX - 1.0) / this.StrideX) + 1 :
                (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;
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

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }

        protected NdArray GetForwardResult(NdArray input, int[] outputIndices, int outputWidth, int outputHeight)
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

            this._outputIndicesList.Add(outputIndices);

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            int[] outputIndices = this._outputIndicesList[this._outputIndicesList.Count - 1];
            this._outputIndicesList.RemoveAt(this._outputIndicesList.Count - 1);

            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (outputIndices[i] != -1)
                {
                    x.Grad[outputIndices[i]] += y.Grad[i];
                }
            }
        }

        public override void ResetState()
        {
            base.ResetState();
            this._outputIndicesList = new List<int[]>();
        }
    }
}
