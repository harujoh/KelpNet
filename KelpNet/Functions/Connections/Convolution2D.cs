using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class Convolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Convolution2D";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _strideX;
        private readonly int _strideY;
        private readonly int _padX;
        private readonly int _padY;

        public readonly int InputCount;
        public readonly int OutputCount;


        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(activation, name, inputNames, outputNames)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._strideX = stride;
            this._strideY = stride;
            this._padX = pad;
            this._padY = pad;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(activation, name, inputNames, outputNames)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this._kWidth = kSize[0];
            this._kHeight = kSize[1];
            this._strideX = stride[0];
            this._strideY = stride[1];
            this._padX = pad[0];
            this._padY = pad[1];
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(Linear linear) : base(linear.Activator, linear.Name, linear.InputNames, linear.OutputNames)
        {
            this._kWidth = 1;
            this._kHeight = 1;
            this._strideX = 1;
            this._strideY = 1;
            this._padX = 0;
            this._padY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Reshape(OutputCount, InputCount, this._kHeight, this._kWidth);
            this.Bias = linear.Bias;
            this.NoBias = linear.NoBias;
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(OutputCount, InputCount, this._kHeight, this._kWidth);
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

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            int outputHeight = (int)Math.Floor((x.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((x.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] y = new Real[x.BatchCount * this.OutputCount * outputHeight * outputWidth];

            for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * this.OutputCount * outputHeight * outputWidth;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    int kOchOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    int yChOffset = yBatchOffset + och * outputHeight * outputWidth;

                    for (int oy = 0; oy < outputHeight * this._strideY; oy += this._strideY)
                    {
                        int iyStart = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int iyLimit = this._kHeight + oy - this._padY < x.Shape[1] ? this._kHeight + oy - this._padY : x.Shape[1];

                        for (int ox = 0; ox < outputWidth * this._strideX; ox += this._strideX)
                        {
                            int ixStart = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int ixLimit = this._kWidth + ox - this._padX < x.Shape[2] ? this._kWidth + ox - this._padX : x.Shape[2];

                            int yIndex = yChOffset + oy / this._strideY * outputWidth + ox / this._strideX;

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                int kIchOffset = kOchOffset + ich * this._kHeight * this._kWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = kIchOffset + (iy - oy + this._padY) * this._kWidth + ix - ox + this._padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        y[yIndex] += x.Data[xIndex] * this.Weight.Data[wIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (this.Activator != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            y[resultIndex] = this.Activator.ForwardActivate(y[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (this.Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = this.Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { this.OutputCount, outputHeight, outputWidth }, x.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgy = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgy.Length; i++)
            {
                activatedgy[i] = this.Activator.BackwardActivate(y.Grad[i], y.Data[i]);
            }

            return activatedgy;
        }

        void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount)
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

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * y.Length;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < y.Shape[0]; och++)
                {
                    int wOchOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = 0; oy < y.Shape[1] * this._strideY; oy += this._strideY)
                    {
                        int iyStart = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int iyLimit = this._kHeight + oy - this._padY < x.Shape[1] ? this._kHeight + oy - this._padY : x.Shape[1];

                        for (int ox = 0; ox < y.Shape[2] * this._strideX; ox += this._strideX)
                        {
                            int ixStart = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int ixLimit = this._kWidth + ox - this._padX < x.Shape[2] ? this._kWidth + ox - this._padX : x.Shape[2];

                            int gyIndex = yBatchOffset + yChOffset + oy / this._strideY * y.Shape[2] + ox / this._strideX;

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                int wIchOffset = wOchOffset + ich * this._kHeight * this._kWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = wIchOffset + (iy - oy + this._padY) * this._kWidth + ix - ox + this._padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

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
