using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Deconvolution2D : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private int _kSize;
        private int _subSample;
        private int _trim;

        public Deconvolution2D(int inputChannels, int outputChannels, int kSize, int subSample = 1, int trim = 0, bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Deconv2D", bool isGpu = true) : base(name,isGpu, inputChannels, outputChannels)
        {
            this._kSize = kSize;
            this._subSample = subSample;
            this._trim = trim;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this.W = new NdArray(outputChannels, inputChannels, kSize, kSize);
            this.gW = NdArray.ZerosLike(this.W);

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                this.W.Data = initialW.Cast<Real>().ToArray();
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = new NdArray(outputChannels);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputSize = (input.Shape[2] - 1) * this._subSample + this._kSize - this._trim * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputSize * outputSize];

            int outSizeOffset = outputSize * outputSize;

            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = this.W.Shape[2] * this.W.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.W.Shape[0]; och++)
                {
                    for (int ich = 0; ich < input.Shape[0]; ich++) //ich = kich input.Shape[0] = this.W.Shape[1]
                    {
                        for (int iy = 0; iy < input.Shape[1]; iy++)
                        {
                            for (int ix = 0; ix < input.Shape[2]; ix++)
                            {
                                int inputIndex = batchCount * input.Length + ich * inputSizeOffset + iy * input.Shape[2] + ix;

                                for (int ky = 0; ky < this.W.Shape[2]; ky++)
                                {
                                    int outIndexY = iy * this._subSample + ky - this._trim;

                                    for (int kx = 0; kx < this.W.Shape[3]; kx++)
                                    {
                                        int outIndexX = ix * this._subSample + kx - this._trim;

                                        int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + outIndexY * outputSize + outIndexX;

                                        int kernelIndex = och * this.W.Shape[1] * kSizeOffset + ich * kSizeOffset + ky * this.W.Shape[3] + kx;

                                        if (outIndexY >= 0 && outIndexY < outputSize && outIndexX >= 0 && outIndexX < outputSize)
                                        {
                                            result[outputIndex] += input.Data[inputIndex] * this.W.Data[kernelIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int oy = 0; oy < outputSize; oy++)
                    {
                        for (int ox = 0; ox < outputSize; ox++)
                        {
                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + oy * outputSize + ox;
                            result[outputIndex] += this.b.Data[och];
                        }
                    }
                }
            }

            return BatchArray.Convert(result, new[] { this.OutputCount, outputSize, outputSize }, input.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            Real[] gx = new Real[prevInput.Data.Length];

            for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.gW.Shape[0]; och++)
                {
                    //Wインデックス用
                    int outChOffset = och * this.gW.Shape[1] * this.gW.Shape[2] * this.gW.Shape[3];

                    //inputインデックス用
                    int inputOffset = och * gy.Shape[1] * gy.Shape[2];

                    for (int ich = 0; ich < this.gW.Shape[1]; ich++)
                    {
                        //Wインデックス用
                        int inChOffset = ich * this.gW.Shape[2] * this.gW.Shape[3];

                        int pinputOffset = ich * prevInput.Shape[1] * prevInput.Shape[2];

                        for (int gwy = 0; gwy < this.gW.Shape[2]; gwy++)
                        {
                            for (int gwx = 0; gwx < this.gW.Shape[3]; gwx++)
                            {
                                for (int py = 0; py < prevInput.Shape[1]; py++)
                                {
                                    int gyy = py * this._subSample + gwy - this._trim;

                                    for (int px = 0; px < prevInput.Shape[2]; px++)
                                    {
                                        int gyx = px * this._subSample + gwx - this._trim;

                                        int gwIndex = outChOffset + inChOffset + gwy * this.gW.Shape[3] + gwx;
                                        int gyIndex = inputOffset + gyy * gy.Shape[2] + gyx + batchCount * gy.Length;

                                        int pInIndex = pinputOffset + py * prevInput.Shape[2] + px + batchCount * prevInput.Length;

                                        if (gyy >= 0 && gyy < gy.Shape[1] &&
                                            gyx >= 0 && gyx < gy.Shape[2])
                                        {
                                            this.gW.Data[gwIndex] += prevInput.Data[pInIndex] * gy.Data[gyIndex];
                                            gx[pInIndex] += this.W.Data[gwIndex] * gy.Data[gyIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int oy = 0; oy < gy.Shape[1]; oy++)
                    {
                        for (int ox = 0; ox < gy.Shape[2]; ox++)
                        {
                            int gyIndex = inputOffset + oy * gy.Shape[2] + ox + batchCount * gy.Length;
                            this.gb.Data[och] += gy.Data[gyIndex];
                        }
                    }
                }
            }

            return BatchArray.Convert(gx, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
