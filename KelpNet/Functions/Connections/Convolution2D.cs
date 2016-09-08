using System;

namespace KelpNet.Functions.Connections
{
    public class Convolution2D : PredictableFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private int _kSize;
        private int _stride;
        private int _pad;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, double[,,,] initialW = null, double[] initialb = null)
        {
            this._kSize = kSize;
            this._stride = stride;
            this._pad = pad;

            this.W = NdArray.Empty(outputChannels, inputChannels, kSize, kSize);
            this.gW = NdArray.ZerosLike(W);

            if (initialW == null)
            {
                InitWeight(W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, W.Data, 0, sizeof(double) * initialW.Length);
            }

            Parameters.Add(new Parameter(this.W, this.gW));

            if (!noBias)
            {
                this.b = NdArray.Zeros(outputChannels);
                this.gb = NdArray.ZerosLike(b);

                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, b.Data, 0, sizeof (double)*initialb.Length);
                }

                Parameters.Add(new Parameter(this.b, this.gb));
            }

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;
        }

        public override NdArray Forward(NdArray input, int batchID = 0)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;

            NdArray result = NdArray.Zeros(OutputCount, outputSize, outputSize);

            NdArray bias = this.b != null ? b : NdArray.Zeros(OutputCount, InputCount);

            for (int i = 0; i < OutputCount; i++)
            {
                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        for (int j = 0; j < InputCount; j++)
                        {
                            for (int dy = 0; dy < this._kSize; dy++)
                            {
                                for (int dx = 0; dx < this._kSize; dx++)
                                {
                                    int inputIndexX = x * this._stride + dx - this._pad;
                                    int inputIndexY = y * this._stride + dy - this._pad;

                                    if (inputIndexY >= 0 && inputIndexY < input.Shape[1] &&
                                        inputIndexX >= 0 && inputIndexX < input.Shape[2]
                                        )
                                    {
                                        result.Data[result.GetIndex(i, y, x)] += input.Get(j, inputIndexY, inputIndexX) * W.Get(i, j, dy, dx);
                                    }
                                }
                            }
                        }

                        result.Data[result.GetIndex(i, y, x)] += bias.Get(i);
                    }
                }
            }

            return result;
        }

        public override NdArray Backward(NdArray gy, NdArray prevInput, NdArray prevOutput, int batchID = 0)
        {
            NdArray gx = NdArray.EmptyLike(prevInput);

            for (int j = 0; j < gy.Shape[0]; j++)
            {
                for (int i = 0; i < prevInput.Shape[0]; i++)
                {
                    for (int y = 0; y < gy.Shape[1]; y++)
                    {
                        for (int x = 0; x < gy.Shape[2]; x++)
                        {
                            for (int dy = 0; dy < this.gW.Shape[2]; dy++)
                            {
                                for (int dx = 0; dx < this.gW.Shape[3]; dx++)
                                {
                                    int prevIndexY = y * this._stride + dy - this._pad;
                                    int prevIndexX = x * this._stride + dx - this._pad;

                                    if (prevIndexY >= 0 && prevIndexY < prevInput.Shape[1] &&
                                        prevIndexX >= 0 && prevIndexX < prevInput.Shape[2])
                                    {
                                        this.gW.Data[gW.GetIndex(j, i, dy, dx)] +=
                                                prevInput.Get(i, prevIndexY, prevIndexX) * gy.Get(j, y, x);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < gx.Shape[0]; i++)
            {
                for (int j = 0; j < gy.Shape[0]; j++)
                {
                    for (int y = 0; y < gy.Shape[1]; y++)
                    {
                        for (int x = 0; x < gy.Shape[2]; x++)
                        {
                            for (int dy = 0; dy < this._kSize; dy++)
                            {
                                for (int dx = 0; dx < this._kSize; dx++)
                                {
                                    int outputIndexY = y * this._stride + dy - this._pad;
                                    int outputIndexX = x * this._stride + dx - this._pad;

                                    if (outputIndexY >= 0 && outputIndexY < gx.Shape[1] &&
                                        outputIndexX >= 0 && outputIndexX < gx.Shape[2]
                                        )
                                    {
                                        gx.Data[gx.GetIndex(i, outputIndexY, outputIndexX)] += W.Get(j, i, dy, dx) *
                                                                                               gy.Data[gy.GetIndex(j, y, x)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (this.b != null)
            {
                for (int i = 0; i < gy.Shape[0]; i++)
                {
                    for (int j = 0; j < gy.Shape[1]; j++)
                    {
                        for (int k = 0; k < gy.Shape[2]; k++)
                        {
                            gb.Data[i] += gy.Get(i, j, k);
                        }
                    }
                }
            }

            return gx;
        }
    }
}
