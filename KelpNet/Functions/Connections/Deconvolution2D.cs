using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Activations;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Deconvolution2D : NeedPreviousInputFunction
    {
        private readonly Activation _activation;
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private int _kWidth;
        private int _kHeight;
        private int _subSample;
        private int _trimX;
        private int _trimY;

        public Deconvolution2D(int inputChannels, int outputChannels, int kSize, int subSample = 1, int trim = 0, bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Deconv2D", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputChannels, outputChannels)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._trimX = trim;
            this._trimY = trim;
            this._subSample = subSample;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];
            this._activation = activation ?? new DummyActivation();

            this.Initialize(initialW, initialb, isGpu);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, Size kSize, int subSample = 1, Size trim = new Size(), bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Deconv2D", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputChannels, outputChannels)
        {
            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._trimX = trim.Width;
            this._trimY = trim.Height;

            this._subSample = subSample;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];
            this._activation = activation ?? new DummyActivation();

            this.Initialize(initialW, initialb, isGpu);
        }

        void Initialize(Real[,,,] initialW = null, Real[] initialb = null, bool isGpu = true)
        {
            this.W = new NdArray(OutputCount, InputCount, this._kHeight, this._kWidth);
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
            this.b = new NdArray(OutputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (this.Parameters.Length > 1)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, "Deconvolution2DForward");
                BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, "Deconvolution2DBackward");
            }
        }

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * this._subSample + this._kHeight - this._trimY * 2;
            int outputWidth = (input.Shape[2] - 1) * this._subSample + this._kWidth - this._trimX * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;

            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = this.W.Shape[2] * this.W.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.OutputCount; och++)
                {
                    for (int oy = this._trimY; oy < outputHeight + this._trimY; oy++)
                    {
                        for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                        {
                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;

                                int kyOffset = oy % this._subSample;
                                for (int ky = kyOffset; ky < this.W.Shape[2]; ky += this._subSample)
                                {
                                    int iy = (oy - ky) / this._subSample;

                                    if (iy >= 0 && iy < input.Shape[1])
                                    {
                                        int kxOffset = ox % this._subSample;
                                        for (int kx = kxOffset; kx < this.W.Shape[3]; kx += this._subSample)
                                        {
                                            int ix = (ox - kx) / this._subSample;

                                            if (ix >= 0 && ix < input.Shape[2])
                                            {
                                                int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                                int kernelIndex = och * this.W.Shape[1] * kSizeOffset + ich * kSizeOffset + ky * this.W.Shape[3] + kx;

                                                result[outputIndex] += input.Data[inputIndex] * this.W.Data[kernelIndex];
                                            }
                                        }
                                    }
                                }
                            }

                            result[outputIndex] += this.b.Data[och];
                        }
                    }
                }
            }

            BatchArray output = BatchArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount);
            if (!(this._activation is DummyActivation))
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (!(this._activation is DummyActivation))
            {
                prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;
                this._prevOutput.RemoveAt(this._prevOutput.Count - 1);
            }

            Real[] gx = new Real[prevInput.Data.Length];

            for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    //Wインデックス用
                    int outChOffset = och * this.gW.Shape[1] * this.gW.Shape[2] * this.gW.Shape[3];

                    //inputインデックス用
                    int inputOffset = och * gy.Shape[1] * gy.Shape[2];

                    for (int oy = this._trimY; oy < gy.Shape[1] + this._trimY; oy++)
                    {
                        for (int ox = this._trimX; ox < gy.Shape[2] + this._trimX; ox++)
                        {
                            int gyIndex = batchCount * gy.Length + inputOffset + (oy - this._trimY) * gy.Shape[2] + ox - this._trimX;
                            Real gyData = gy.Data[gyIndex];
                            this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

                            for (int ich = 0; ich < InputCount; ich++) // InputCount = this.gW.Shape[1] = prevInput.Shape[0]
                            {
                                //Wインデックス用
                                int inChOffset = ich * this.gW.Shape[2] * this.gW.Shape[3];
                                int pinputOffset = ich * prevInput.Shape[1] * prevInput.Shape[2];

                                int gwyOffset = oy % this._subSample;
                                for (int gwy = gwyOffset; gwy < this.gW.Shape[2]; gwy += this._subSample)
                                {
                                    int py = (oy - gwy) / this._subSample;

                                    if (py >= 0 && py < prevInput.Shape[1])
                                    {
                                        int gwxOffset = ox % this._subSample;
                                        for (int gwx = gwxOffset; gwx < this.gW.Shape[3]; gwx += this._subSample)
                                        {
                                            int px = (ox - gwx) / this._subSample;

                                            if (px >= 0 && px < prevInput.Shape[2])
                                            {
                                                int pInIndex = batchCount * prevInput.Length + pinputOffset + py * prevInput.Shape[2] + px;
                                                int gwIndex = outChOffset + inChOffset + gwy * this.gW.Shape[3] + gwx;

                                                this.gW.Data[gwIndex] += prevInput.Data[pInIndex] * gyData;
                                                gx[pInIndex] += this.W.Data[gwIndex] * gyData;
                                            }
                                        }
                                    }
                                }
                            }

                            this.gb.Data[och] += gyData;
                        }
                    }
                }
            }

            return BatchArray.Convert(gx, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
