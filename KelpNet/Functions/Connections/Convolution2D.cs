using System;
using KelpNet.Common;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly int _kSize;
        private readonly int _stride;
        private readonly int _pad;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, double[,,,] initialW = null, double[] initialb = null, string name = "Conv2D", bool isParallel = true) : base(name, inputChannels, outputChannels, isParallel)
        {
            this._kSize = kSize;
            this._stride = stride;
            this._pad = pad;

            this.W = NdArray.Zeros(outputChannels, inputChannels, kSize, kSize);
            this.gW = NdArray.ZerosLike(this.W);

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //サイズチェックを兼ねる
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputChannels);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;

            double[] result = new double[this.OutputCount * outputSize * outputSize];
            int resultIndex = 0;

            for (int och = 0; och < this.OutputCount; och++)
            {
                //Wインデックス用
                int outChOffset = och * this.InputCount * this._kSize * this._kSize;

                for (int oy = 0; oy < outputSize; oy++)
                {
                    for (int ox = 0; ox < outputSize; ox++)
                    {
                        for (int ich = 0; ich < input.Shape[0]; ich++)
                        {
                            //Wインデックス用
                            int inChOffset = ich * this._kSize * this._kSize;

                            //inputインデックス用
                            int inputOffset = ich * input.Shape[1] * input.Shape[2];

                            for (int ky = 0; ky < this._kSize; ky++)
                            {
                                int iy = oy * this._stride + ky - this._pad;

                                if (iy >= 0 && iy < input.Shape[1])
                                {
                                    for (int kx = 0; kx < this._kSize; kx++)
                                    {
                                        int ix = ox * this._stride + kx - this._pad;

                                        if (ix >= 0 && ix < input.Shape[2])
                                        {
                                            int wIndex = outChOffset + inChOffset + ky * this._kSize + kx;
                                            int inputIndex = inputOffset + iy * input.Shape[2] + ix;

                                            result[resultIndex] += input.Data[inputIndex] * this.W.Data[wIndex];
                                        }
                                    }
                                }
                            }
                        }

                        result[resultIndex] += this.b.Data[och];
                        resultIndex++;
                    }
                }
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputSize, outputSize });
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput)
        {
            double[] gx = new double[prevInput.Length];

            int gyIndex = 0;

            for (int och = 0; och < gy.Shape[0]; och++)
            {
                //gWインデックス用
                int outChOffset = och * this.InputCount * this._kSize * this._kSize;

                for (int oy = 0; oy < gy.Shape[1]; oy++)
                {
                    for (int ox = 0; ox < gy.Shape[2]; ox++)
                    {
                        double gyData = gy.Data[gyIndex++]; //gyIndex = ch * x * y

                        for (int ich = 0; ich < prevInput.Shape[0]; ich++)
                        {
                            //gWインデックス用
                            int inChOffset = ich * this._kSize * this._kSize;

                            //inputインデックス用
                            int inputOffset = ich * prevInput.Shape[1] * prevInput.Shape[2];

                            for (int ky = 0; ky < this._kSize; ky++)
                            {
                                int iy = oy * this._stride + ky - this._pad;

                                if (iy >= 0 && iy < prevInput.Shape[1])
                                {
                                    for (int kx = 0; kx < this._kSize; kx++)
                                    {
                                        int ix = ox * this._stride + kx - this._pad;

                                        if (ix >= 0 && ix < prevInput.Shape[2])
                                        {
                                            //WとgWのshapeは等しい
                                            int wIndex = outChOffset + inChOffset + ky * this._kSize + kx;

                                            //prevInputとgxのshapeは等しい
                                            int inputIndex = inputOffset + iy * prevInput.Shape[2] + ix;

                                            this.gW.Data[wIndex] += prevInput.Data[inputIndex] * gyData;

                                            gx[inputIndex] += this.W.Data[wIndex] * gyData;
                                        }
                                    }
                                }
                            }
                        }

                        this.gb.Data[och] += gyData;
                    }
                }
            }

            return NdArray.Convert(gx, prevInput.Shape);
        }
    }
}
