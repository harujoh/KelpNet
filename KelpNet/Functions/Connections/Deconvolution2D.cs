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
                        int iyLimit = oy / this._subSample + 1 < input.Shape[1] ? oy / this._subSample + 1 : input.Shape[1];
                        int iyStart = oy - this.W.Shape[3] < 0 ? 0 : (oy - this.W.Shape[2]) / this._subSample + 1;

                        for (int ox = this._trimX; ox < outputWidth + this._trimX; ox++)
                        {
                            int ixLimit = ox / this._subSample + 1 < input.Shape[2] ? ox / this._subSample + 1 : input.Shape[2];
                            int ixStart = ox - this.W.Shape[3] < 0 ? 0 : (ox - this.W.Shape[3]) / this._subSample + 1;

                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._trimY) * outputWidth + ox - this._trimX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = och * this.W.Shape[1] * kSizeOffset + ich * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * this._subSample) * this.W.Shape[3] + (ox - ix * this._subSample);

                                        result[outputIndex] += input.Data[inputIndex] * this.W.Data[kernelIndex];
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
                    int outChOffset = och * this.gW.Shape[1] * this.gW.Shape[2] * this.gW.Shape[3];
                    int inputOffset = och * gy.Shape[1] * gy.Shape[2];

                    for (int oy = this._trimY; oy < gy.Shape[1] + this._trimY; oy++)
                    {
                        int iyLimit = oy / this._subSample + 1 < prevInput.Shape[1] ? oy / this._subSample + 1 : prevInput.Shape[1];
                        int iyStart = oy - this.W.Shape[3] < 0 ? 0 : (oy - this.W.Shape[2]) / this._subSample + 1;

                        for (int ox = this._trimX; ox < gy.Shape[2] + this._trimX; ox++)
                        {
                            int ixLimit = ox / this._subSample + 1 < prevInput.Shape[2] ? ox / this._subSample + 1 : prevInput.Shape[2];
                            int ixStart = ox - this.W.Shape[3] < 0 ? 0 : (ox - this.W.Shape[3]) / this._subSample + 1;

                            int gyIndex = batchCount * gy.Length + inputOffset + (oy - this._trimY) * gy.Shape[2] + ox - this._trimX;
                            Real gyData = gy.Data[gyIndex];
                            this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

                            for (int ich = 0; ich < InputCount; ich++) // InputCount = this.gW.Shape[1] = prevInput.Shape[0]
                            {
                                int inChOffset = outChOffset + ich * this.gW.Shape[2] * this.gW.Shape[3];
                                int pinputOffset = batchCount * prevInput.Length + ich * prevInput.Shape[1] * prevInput.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int pInIndex = pinputOffset + iy * prevInput.Shape[2] + ix;
                                        int gwIndex = inChOffset + (oy - iy * this._subSample) * this.gW.Shape[3] + (ox - ix * this._subSample);

                                        this.gW.Data[gwIndex] += prevInput.Data[pInIndex] * gyData;
                                        gx[pInIndex] += this.W.Data[gwIndex] * gyData;
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
