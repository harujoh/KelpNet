using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class Convolution2D<T> : CompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Convolution2D";
        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "localResult = ForwardActivate(localResult);";

        public NdArray<T> Weight;
        public NdArray<T> Bias;

        public readonly bool NoBias;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _strideX;
        private readonly int _strideY;
        private readonly int _padX;
        private readonly int _padY;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._strideX = stride;
            this._strideY = stride;
            this._padX = pad;
            this._padY = pad;
            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, int kWidth, int kHeight, int strideX=1, int strideY=1, int padX=0, int padY=0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames)
        {
            this._kWidth = kWidth;
            this._kHeight = kHeight;
            this._strideX = strideX;
            this._strideY = strideY;
            this._padX = padX;
            this._padY = padY;
            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(Linear<T> linear) : base(FUNCTION_NAME, linear.Activator, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, linear.Name, linear.InputNames, linear.OutputNames)
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
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real<T>[] result = new Real<T>[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
            {
                int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    //Wインデックス用
                    int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    for (int oy = 0; oy < outputHeight * this._strideY; oy += this._strideY)
                    {
                        int kyStartIndex = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int kyLimit = this._kHeight + oy - this._padY < input.Shape[1] ? this._kHeight + oy - this._padY : input.Shape[1];

                        for (int ox = 0; ox < outputWidth * this._strideX; ox += this._strideX)
                        {
                            int kxStartIndex = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int kxLimit = this._kWidth + ox - this._padX < input.Shape[2] ? this._kWidth + ox - this._padX : input.Shape[2];

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                //Wインデックス用
                                int inChOffset = ich * this._kHeight * this._kWidth;

                                //inputインデックス用
                                int inputOffset = ich * input.Shape[1] * input.Shape[2];

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        int wIndex = outChOffset + inChOffset + (ky - oy + this._padY) * this._kWidth + kx - ox + this._padX;
                                        int inputIndex = inputOffset + ky * input.Shape[2] + kx + batchCounter * input.Length;

                                        result[resultIndex] += input.Data[inputIndex] * this.Weight.Data[wIndex];
                                    }
                                }
                            }

                            resultIndex++;
                        }
                    }
                }
            }

            if (this.Activator != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += this.Bias.Data[och];
                            result[resultIndex] = this.Activator.ForwardActivate(result[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += this.Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (this.Activator != null)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] = this.Activator.ForwardActivate(result[resultIndex]);
                            resultIndex++;
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
            Real<T>[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    //gWインデックス用
                    int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    for (int oy = 0; oy < y.Shape[1] * this._strideY; oy += this._strideY)
                    {
                        //計算省略のためにジャンプ
                        int kyStartIndex = this._padY - oy < 0 ? 0 : this._padY - oy;
                        int kyLimit = this._kHeight < x.Shape[1] - oy + this._padY ? this._kHeight : x.Shape[1] - oy + this._padY;

                        for (int ox = 0; ox < y.Shape[2] * this._strideX; ox += this._strideX)
                        {
                            //計算省略のためにジャンプ
                            int kxStartIndex = this._padX - ox < 0 ? 0 : this._padX - ox;
                            int kxLimit = this._kWidth < x.Shape[2] - ox + this._padX ? this._kWidth : x.Shape[2] - ox + this._padX;

                            int gyIndex = batchCounter * y.Length + och * y.Shape[1] * y.Shape[2] + oy * y.Shape[2] + ox;

                            Real<T> gyData = activatedgy[gyIndex];

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                //gWインデックス用
                                int inChOffset = ich * this._kHeight * this._kWidth;

                                //inputインデックス用
                                int inputOffset = ich * x.Shape[1] * x.Shape[2] + batchCounter * x.Length;

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        //WとgWのshapeは等しい
                                        int wIndex = outChOffset + inChOffset + ky * this._kWidth + kx;

                                        //xとgxのshapeは等しい
                                        int inputIndex = inputOffset + (ky + oy - this._padY) * x.Shape[2] + kx + ox - this._padX;

                                        this.Weight.Grad[wIndex] += x.Data[inputIndex] * gyData;

                                        x.Grad[inputIndex] += this.Weight.Data[wIndex] * gyData;
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
