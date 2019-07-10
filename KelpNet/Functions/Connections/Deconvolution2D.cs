using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class Deconvolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Deconvolution2D";

        public NdArray Weight;
        public NdArray Bias;

        public bool NoBias;

        public int KernelWidth;
        public int KernelHeight;
        public int StrideX;
        public int StrideY;
        public int PadX;
        public int PadY;

        public int InputCount;
        public int OutputCount;

        public Deconvolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(activation, name, inputNames, outputNames)
        {
            this.KernelWidth = kernelSize;
            this.KernelHeight = kernelSize;
            this.PadX = pad;
            this.PadY = pad;
            this.StrideX = stride;
            this.StrideY = stride;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] subSample = null, int[] trim = null, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(activation, name, inputNames, outputNames)
        {
            if (subSample == null)
                subSample = new[] { 1, 1 };

            if (trim == null)
                trim = new[] { 0, 0 };

            this.KernelWidth = kernelSize[0];
            this.KernelHeight = kernelSize[1];
            this.PadX = trim[0];
            this.PadY = trim[1];
            this.NoBias = noBias;

            this.StrideX = subSample[0];
            this.StrideY = subSample[1];

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(InputCount, OutputCount, this.KernelHeight, this.KernelWidth);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters[0] = this.Weight;


            if (!NoBias)
            {
                this.Bias = new NdArray(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.ToRealArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        public override NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * this.StrideY + this.KernelHeight - this.PadY * 2;
            int outputWidth = (input.Shape[2] - 1) * this.StrideX + this.KernelWidth - this.PadX * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;
            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = this.Weight.Shape[2] * this.Weight.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.OutputCount; och++)
                {
                    for (int oy = this.PadY; oy < outputHeight + this.PadY; oy++)
                    {
                        int iyLimit = oy / this.StrideY + 1 < input.Shape[1] ? oy / this.StrideY + 1 : input.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this.StrideY + 1;

                        for (int ox = this.PadX; ox < outputWidth + this.PadX; ox++)
                        {
                            int ixLimit = ox / this.StrideX + 1 < input.Shape[2] ? ox / this.StrideX + 1 : input.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this.StrideX + 1;

                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this.PadY) * outputWidth + ox - this.PadX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = ich * this.Weight.Shape[1] * kSizeOffset + och * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * this.StrideY) * this.Weight.Shape[3] + (ox - ix * this.StrideX);

                                        result[outputIndex] += input.Data[inputIndex] * this.Weight.Data[kernelIndex];
                                    }
                                }
                            }

                        }
                    }
                }
            }

            if (this.Activator != null && !NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this.PadY; oy < outputHeight + this.PadY; oy++)
                        {
                            for (int ox = this.PadX; ox < outputWidth + this.PadX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this.PadY) * outputWidth + ox - this.PadX;

                                result[outputIndex] += this.Bias.Data[och];
                                result[outputIndex] = this.Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this.PadY; oy < outputHeight + this.PadY; oy++)
                        {
                            for (int ox = this.PadX; ox < outputWidth + this.PadX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this.PadY) * outputWidth + ox - this.PadX;

                                result[outputIndex] += this.Bias.Data[och];
                            }
                        }
                    }
                }
            }
            else if (this.Activator != null)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this.PadY; oy < outputHeight + this.PadY; oy++)
                        {
                            for (int ox = this.PadX; ox < outputWidth + this.PadX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this.PadY) * outputWidth + ox - this.PadX;

                                result[outputIndex] = this.Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected Real[] GetActivatedgy(NdArray y)
        {
            int gyIndex = 0;

            Real[] activatedgy = new Real[y.Grad.Length];

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    for (int olocation = 0; olocation < y.Shape[1] * y.Shape[2]; olocation++)
                    {
                        activatedgy[gyIndex] = this.Activator.BackwardActivate(y.Grad[gyIndex], y.Data[gyIndex]);
                        gyIndex++;
                    }
                }
            }

            return activatedgy;
        }

        protected void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        this.Bias.Grad[och] += gy[gyIndex];
                        gyIndex++;
                    }
                }
            }
        }

        public override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    int wOchOffset = och * this.Weight.Shape[2] * this.Weight.Shape[3];
                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = this.PadY; oy < y.Shape[1] + this.PadY; oy++)
                    {
                        int iyLimit = oy / this.StrideY + 1 < x.Shape[1] ? oy / this.StrideY + 1 : x.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this.StrideY + 1;

                        for (int ox = this.PadX; ox < y.Shape[2] + this.PadX; ox++)
                        {
                            int ixLimit = ox / this.StrideX + 1 < x.Shape[2] ? ox / this.StrideX + 1 : x.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this.StrideX + 1;

                            int gyIndex = batchCount * y.Length + yChOffset + (oy - this.PadY) * y.Shape[2] + ox - this.PadX;

                            for (int ich = 0; ich < InputCount; ich++)
                            {
                                int wIchOffset = ich * this.Weight.Shape[1] * this.Weight.Shape[2] * this.Weight.Shape[3] + wOchOffset;
                                int xChOffset = batchCount * x.Length + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;
                                        int wIndex = wIchOffset + (oy - iy * this.StrideY) * this.Weight.Shape[3] + (ox - ix * this.StrideX);

                                        this.Weight.Grad[wIndex] += x.Data[xIndex] * activatedgy[gyIndex];
                                        x.Grad[xIndex] += this.Weight.Data[wIndex] * activatedgy[gyIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
