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

        private int _kSize;
        private int _stride;
        private int _pad;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, double[,,,] initialW = null, double[] initialb = null, string name = "Conv2D") : base(name, inputChannels, outputChannels)
        {
            this._kSize = kSize;
            this._stride = stride;
            this._pad = pad;

            this.W = NdArray.Zeros(outputChannels, inputChannels, kSize, kSize);
            this.gW = NdArray.ZerosLike(this.W);

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //サイズチェックを兼ねる
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters.Add(new OptimizeParameter(this.W, this.gW, this.Name + " W"));

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputChannels);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                this.Parameters.Add(new OptimizeParameter(this.b, this.gb, this.Name + " b"));
            }
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;

            double[] result = new double[this.OutputCount * outputSize * outputSize];
            int resultIndex = 0;

            for (int i = 0; i < this.OutputCount; i++)
            {
                //Wインデックス用
                int outChOffset = i * this.InputCount * this._kSize * this._kSize;

                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        for (int j = 0; j < input.Shape[0]; j++)
                        {
                            //Wインデックス用
                            int inChOffset = j * this._kSize * this._kSize;

                            //inputインデックス用
                            int inputOffset = j * input.Shape[1] * input.Shape[2];

                            for (int dy = 0; dy < this._kSize; dy++)
                            {
                                int inputIndexY = y * this._stride + dy - this._pad;

                                if (inputIndexY >= 0 && inputIndexY < input.Shape[1])
                                {
                                    for (int dx = 0; dx < this._kSize; dx++)
                                    {
                                        int inputIndexX = x * this._stride + dx - this._pad;

                                        if (inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                        {
                                            int wIndex = outChOffset + inChOffset + dy * this._kSize + dx;
                                            int inputIndex = inputOffset + inputIndexY * input.Shape[2] + inputIndexX;

                                            result[resultIndex] += input.Data[inputIndex] * this.W.Data[wIndex];
                                        }
                                    }
                                }
                            }
                        }

                        result[resultIndex] += this.b.Data[i];
                        resultIndex++;
                    }
                }
            }

            return new NdArray(result, new[] { this.OutputCount, outputSize, outputSize });
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput)
        {
            double[] gx = new double[prevInput.Length];

            int gyIndex = 0;

            for (int i = 0; i < gy.Shape[0]; i++)
            {
                //gWインデックス用
                int outChOffset = i * this.InputCount * this._kSize * this._kSize;

                for (int y = 0; y < gy.Shape[1]; y++)
                {
                    for (int x = 0; x < gy.Shape[2]; x++)
                    {
                        double gyData = gy.Data[gyIndex];

                        for (int j = 0; j < prevInput.Shape[0]; j++)
                        {
                            //gWインデックス用
                            int inChOffset = j * this._kSize * this._kSize;

                            //inputインデックス用
                            int inputOffset = j * prevInput.Shape[1] * prevInput.Shape[2];

                            for (int dy = 0; dy < this._kSize; dy++)
                            {
                                int indexY = y * this._stride + dy - this._pad;

                                if (indexY >= 0 && indexY < prevInput.Shape[1])
                                {
                                    for (int dx = 0; dx < this._kSize; dx++)
                                    {
                                        int indexX = x * this._stride + dx - this._pad;

                                        if (indexX >= 0 && indexX < prevInput.Shape[2])
                                        {
                                            //WとgWのshapeは等しい
                                            int wIndex = outChOffset + inChOffset + dy * this._kSize + dx;

                                            //prevInputとgxのshapeは等しい
                                            int inputIndex = inputOffset + indexY * prevInput.Shape[2] + indexX;

                                            this.gW.Data[wIndex] += prevInput.Data[inputIndex] * gyData;

                                            gx[inputIndex] += this.W.Data[wIndex] * gyData;
                                        }
                                    }
                                }
                            }
                        }

                        this.gb.Data[i] += gyData;
                        gyIndex++;
                    }
                }
            }

            return new NdArray(gx, prevInput.Shape);
        }
    }
}
