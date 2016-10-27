using System;
using KelpNet.Common;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousDataFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private int _kSize;
        private int _stride;
        private int _pad;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, double[,,,] initialW = null, double[] initialb = null, string name = "Conv2D") : base(name)
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

            Parameters.Add(new OptimizeParameter(this.W, this.gW, Name + " W"));


            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputChannels);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                Parameters.Add(new OptimizeParameter(this.b, this.gb, Name + " b"));
            }

            OutputCount = outputChannels;
            InputCount = inputChannels;
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;

            double[] result = new double[OutputCount * outputSize * outputSize];
            int resultIndex = 0;

            for (int j = 0; j < OutputCount; j++)
            {
                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        for (int k = 0; k < input.Shape[0]; k++)
                        {
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
                                            result[resultIndex] +=
                                                input.Get(k, inputIndexY, inputIndexX) * this.W.Get(j, k, dy, dx);
                                        }
                                    }
                                }
                            }
                        }

                        result[resultIndex] += this.b.Data[j];
                        resultIndex++;
                    }
                }
            }

            return new NdArray(result, new[] { OutputCount, outputSize, outputSize });
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[prevInput.Length];

            int gyIndex = 0;

            for (int k = 0; k < gy.Shape[0]; k++)
            {
                for (int y = 0; y < gy.Shape[1]; y++)
                {
                    for (int x = 0; x < gy.Shape[2]; x++)
                    {
                        double gyData = gy.Data[gyIndex];

                        for (int j = 0; j < prevInput.Shape[0]; j++)
                        {
                            for (int dy = 0; dy < this._kSize; dy++)
                            {
                                int indexY = y * this._stride + dy - this._pad;

                                if (indexY >= 0 && indexY < prevInput.Shape[1])
                                {
                                    for (int dx = 0; dx < this._kSize; dx++)
                                    {
                                        int indexX = x * this._stride + dx - this._pad;

                                        //prevInputとgxのshapeは等しい
                                        int index = prevInput.GetIndex(j, indexY, indexX);

                                        //WとgWのshapeは等しい
                                        int wIndex = this.W.GetIndex(k, j, dy, dx);

                                        if (indexX >= 0 && indexX < prevInput.Shape[2])
                                        {
                                            this.gW.Data[wIndex] += prevInput.Data[index] * gyData;

                                            gx[index] += this.W.Data[wIndex] * gyData;
                                        }
                                    }
                                }
                            }
                        }

                        this.gb.Data[k] += gyData;
                        gyIndex++;
                    }
                }
            }

            return new NdArray(gx, prevInput.Shape);
        }
    }
}
