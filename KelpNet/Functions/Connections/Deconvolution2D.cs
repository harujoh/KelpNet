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
            int outputWidth = (input.Shape[2] - 1) * this._subSample + this._kWidth - this._trimX * 2;
            int outputHeight = (input.Shape[2] - 1) * this._subSample + this._kHeight - this._trimY * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;

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
                                    int outIndexY = iy * this._subSample + ky - this._trimY;

                                    for (int kx = 0; kx < this.W.Shape[3]; kx++)
                                    {
                                        int outIndexX = ix * this._subSample + kx - this._trimX;

                                        int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + outIndexY * outputWidth + outIndexX;

                                        int kernelIndex = och * this.W.Shape[1] * kSizeOffset + ich * kSizeOffset + ky * this.W.Shape[3] + kx;

                                        if (outIndexY >= 0 && outIndexY < outputHeight && outIndexX >= 0 && outIndexX < outputWidth)
                                        {
                                            result[outputIndex] += input.Data[inputIndex] * this.W.Data[kernelIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        for (int ox = 0; ox < outputWidth; ox++)
                        {
                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + oy * outputWidth + ox;
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
                                    int gyy = py * this._subSample + gwy - this._trimY;

                                    for (int px = 0; px < prevInput.Shape[2]; px++)
                                    {
                                        int gyx = px * this._subSample + gwx - this._trimX;

                                        int gwIndex = outChOffset + inChOffset + gwy * this.gW.Shape[3] + gwx;
                                        int gyIndex = inputOffset + gyy * gy.Shape[2] + gyx + batchCount * gy.Length;

                                        int pInIndex = pinputOffset + py * prevInput.Shape[2] + px + batchCount * prevInput.Length;

                                        if (gyy >= 0 && gyy < gy.Shape[1] &&
                                            gyx >= 0 && gyx < gy.Shape[2])
                                        {
                                            Real gyData = gy.Data[gyIndex]; //gyIndex = ch * ox * oy
                                            this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

                                            this.gW.Data[gwIndex] += prevInput.Data[pInIndex] * gyData;
                                            gx[pInIndex] += this.W.Data[gwIndex] * gyData;
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
                            Real gyData = gy.Data[gyIndex];
                            this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

                            this.gb.Data[och] += gyData;
                        }
                    }
                }
            }

            return BatchArray.Convert(gx, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
