using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class Deconvolution2D<T> : CompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Deconvolution2D";
        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "result = ForwardActivate(result);";

        public NdArray<T> Weight;
        public NdArray<T> Bias;

        public readonly bool NoBias;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _subSampleX;
        private readonly int _subSampleY;
        private readonly int _trimX;
        private readonly int _trimY;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Deconvolution2D(int inputChannels, int outputChannels, int kSize, int subSample = 1, int trim = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, activation, new []{new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE)}, name, inputNames, outputNames)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._trimX = trim;
            this._trimY = trim;
            this._subSampleX = subSample;
            this._subSampleY = subSample;
            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int kWidth, int kHeight, int subSampleX = 1, int subSampleY = 1, int trimX = 0, int trimY = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames)
        {
            this._kWidth = kWidth;
            this._kHeight = kHeight;
            this._trimX = trimX;
            this._trimY = trimY;
            this.NoBias = noBias;

            this._subSampleX = subSampleX;
            this._subSampleY = subSampleY;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray<T>(OutputCount, InputCount, this._kHeight, this._kWidth);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer<T>.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real<T>.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;


            if (!NoBias)
            {
                this.Bias = new NdArray<T>(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real<T>.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        protected override NdArray<T> NeedPreviousForwardCpu(NdArray<T> input)
        {
            int outputHeight = (input.Shape[1] - 1) * this._subSampleY + this._kHeight - this._trimY * 2;
            int outputWidth = (input.Shape[2] - 1) * this._subSampleX + this._kWidth - this._trimX * 2;

            Real<T>[] result = new Real<T>[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;
            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = this.Weight.Shape[2] * this.Weight.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.OutputCount; och++)
                {
                    for (int oy = this._trimY; oy < outputHeight + this._trimY; oy++)
                    {
                        int iyLimit = oy / this._subSampleY + 1 < input.Shape[1] ? oy / this._subSampleY + 1 : input.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this._subSampleY + 1;

                        for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                        {
                            int ixLimit = ox / this._subSampleX + 1 < input.Shape[2] ? ox / this._subSampleX + 1 : input.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this._subSampleX + 1;

                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = och * this.Weight.Shape[1] * kSizeOffset + ich * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * this._subSampleY) * this.Weight.Shape[3] + (ox - ix * this._subSampleX);

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
                        for (int oy = this._trimY; oy < outputHeight + this._trimY; oy++)
                        {
                            for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

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
                        for (int oy = this._trimY; oy < outputHeight + this._trimY; oy++)
                        {
                            for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

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
                        for (int oy = this._trimY; oy < outputHeight + this._trimY; oy++)
                        {
                            for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

                                result[outputIndex] = this.Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }

            return NdArray<T>.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        Real<T>[] GetActivatedgy(NdArray<T> y)
        {
            int gyIndex = 0;

            Real<T>[] activatedgy = new Real<T>[y.Grad.Length];

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

        void CalcBiasGrad(Real<T>[] gy, int[] gyShape, int batchCount)
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

        protected override void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            //Real<T>[] gx = new Real<T>[x.Data.Length];
            Real<T>[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            //本来のロジック
            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    int outChOffset = och * this.Weight.Shape[1] * this.Weight.Shape[2] * this.Weight.Shape[3];
                    int inputOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = this._trimY; oy < y.Shape[1] + this._trimY; oy++)
                    {
                        int iyLimit = oy / this._subSampleY + 1 < x.Shape[1] ? oy / this._subSampleY + 1 : x.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this._subSampleY + 1;

                        for (int ox = this._trimX; ox < y.Shape[2] + this._trimX; ox++)
                        {
                            int ixLimit = ox / this._subSampleX + 1 < x.Shape[2] ? ox / this._subSampleX + 1 : x.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this._subSampleX + 1;

                            int gyIndex = batchCount * y.Length + inputOffset + (oy - this._trimY) * y.Shape[2] + ox - this._trimX;
                            Real<T> gyData = activatedgy[gyIndex];

                            for (int ich = 0; ich < InputCount; ich++)
                            {
                                int inChOffset = outChOffset + ich * this.Weight.Shape[2] * this.Weight.Shape[3];
                                int pinputOffset = batchCount * x.Length + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int pInIndex = pinputOffset + iy * x.Shape[2] + ix;
                                        int gwIndex = inChOffset + (oy - iy * this._subSampleY) * this.Weight.Shape[3] + (ox - ix * this._subSampleX);

                                        this.Weight.Grad[gwIndex] += x.Data[pInIndex] * gyData;
                                        x.Grad[pInIndex] += this.Weight.Data[gwIndex] * gyData;
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
