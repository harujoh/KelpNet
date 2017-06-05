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
                    for (int ich = 0; ich < input.Shape[0]; ich++)
                    {
                        for (int iy = 0; iy < input.Shape[1]; iy++)
                        {
                            int kyOffset = iy * this._subSample - this._trimY;
                            int kyStartOffset = kyOffset < 0 ? 0 : kyOffset;
                            int kyLimit = this.W.Shape[2] + kyOffset < outputHeight ? this.W.Shape[2] + kyOffset : outputHeight;

                            for (int ix = 0; ix < input.Shape[2]; ix++)
                            {
                                int kxOffset = ix * this._subSample - this._trimX;
                                int kxStartOffset = kxOffset < 0 ? 0 : kxOffset;
                                int kxLimit = this.W.Shape[3] + kxOffset < outputWidth ? this.W.Shape[3] + kxOffset : outputWidth;

                                int inputIndex = batchCount * input.Length + ich * inputSizeOffset + iy * input.Shape[2] + ix;

                                for (int ky = kyStartOffset; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartOffset; kx < kxLimit; kx++)
                                    {
                                        int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + ky * outputWidth + kx;
                                        int kernelIndex = och * this.W.Shape[1] * kSizeOffset + ich * kSizeOffset + (ky - kyOffset) * this.W.Shape[3] + kx - kxOffset;

                                        result[outputIndex] += input.Data[inputIndex] * this.W.Data[kernelIndex];
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

                        for (int py = 0; py < prevInput.Shape[1]; py++)
                        {
                            int gwyOffset = py * this._subSample - this._trimY;
                            int gwyStartOffset = gwyOffset < 0 ? 0 : gwyOffset;
                            int gwyLimit = this.gW.Shape[2] + gwyOffset < gy.Shape[1] ? this.gW.Shape[2] + gwyOffset : gy.Shape[1];

                            for (int px = 0; px < prevInput.Shape[2]; px++)
                            {
                                int gwxOffset = px * this._subSample - this._trimX;
                                int gwxStartOffset = gwxOffset < 0 ? 0 : gwxOffset;
                                int gwxLimit = this.gW.Shape[3] + gwxOffset < gy.Shape[2] ? this.gW.Shape[3] + gwxOffset : gy.Shape[2];

                                for (int gwy = gwyStartOffset; gwy < gwyLimit; gwy++)
                                {
                                    for (int gwx = gwxStartOffset; gwx < gwxLimit; gwx++)
                                    {
                                        int pInIndex = batchCount * prevInput.Length + pinputOffset + py * prevInput.Shape[2] + px;

                                        int gwIndex = outChOffset + inChOffset + (gwy - gwyOffset) * this.gW.Shape[3] + gwx - gwxOffset;
                                        int gyIndex = batchCount * gy.Length + inputOffset + gwy * gy.Shape[2] + gwx;

                                        Real gyData = gy.Data[gyIndex];
                                        this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

                                        this.gW.Data[gwIndex] += prevInput.Data[pInIndex] * gyData;
                                        gx[pInIndex] += this.W.Data[gwIndex] * gyData;
                                    }
                                }
                            }
                        }
                    }

                    for (int oy = 0; oy < gy.Shape[1]; oy++)
                    {
                        for (int ox = 0; ox < gy.Shape[2]; ox++)
                        {
                            int gyIndex = batchCount * gy.Length + inputOffset + oy * gy.Shape[2] + ox;
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
