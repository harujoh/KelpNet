using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousInputFunction
    {
        const string FUNCTION_NAME = "Convolution2D";

        private readonly Activation _activation;
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardgWKernel;

        [NonSerialized]
        public ComputeKernel BackwardgXKernel;

        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _strideX;
        private readonly int _strideY;
        private readonly int _padX;
        private readonly int _padY;

        public bool IsGpu;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool isGpu = false, Activation activation = null) : base(name, inputChannels, outputChannels)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._strideX = stride;
            this._strideY = stride;
            this._padX = pad;
            this._padY = pad;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this._activation = activation;

            this.IsGpu = isGpu && Weaver.Enable;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, Size stride = new Size(), Size pad = new Size(), bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool isGpu = false, Activation activation = null) : base(name, inputChannels, outputChannels)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(0, 0);

            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._strideX = stride.Width;
            this._strideY = stride.Height;
            this._padX = pad.Width;
            this._padY = pad.Height;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this._activation = activation;

            this.IsGpu = isGpu && Weaver.Enable;

            this.Initialize(initialW, initialb);
        }

        void Initialize(Array initialW = null, Array initialb = null)
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
                    this.b.Data = initialb.Cast<Real>().ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            if (IsGpu)
            {
                var KernelSource = Weaver.GetKernelSource(FUNCTION_NAME);

                if (this._activation != null)
                {
                    KernelSource = this._activation.ActivateFunctionString + KernelSource.Replace("/*ForwardActivate*/", "ForwardActivate(gpuY + index);");
                }

                var program = Weaver.CreateProgram(KernelSource);
                this.ForwardKernel = program.CreateKernel("Convolution2DForward");
                this.BackwardgWKernel = program.CreateKernel("Convolution2DgWBackward");
                this.BackwardgXKernel = program.CreateKernel("Convolution2DgXBackward");
            }
        }

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            if (!IsGpu)
            {
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

                                            result[resultIndex] += input.Data[inputIndex] * this.W.Data[wIndex];
                                        }
                                    }
                                }

                                result[resultIndex] += this.b.Data[och];
                                if (this._activation != null) this._activation.ForwardActivate(ref result[resultIndex]);
                                resultIndex++;
                            }
                        }
                    }
                }
            }
            else
            {
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.b.Data))
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
                {
                    ForwardKernel.SetMemoryArgument(0, gpuX);
                    ForwardKernel.SetMemoryArgument(1, gpuW);
                    ForwardKernel.SetMemoryArgument(2, gpub);
                    ForwardKernel.SetMemoryArgument(3, gpuY);
                    ForwardKernel.SetValueArgument(4, input.Shape[1]);
                    ForwardKernel.SetValueArgument(5, input.Shape[2]);
                    ForwardKernel.SetValueArgument(6, input.Length);
                    ForwardKernel.SetValueArgument(7, outputWidth);
                    ForwardKernel.SetValueArgument(8, outputHeight);
                    ForwardKernel.SetValueArgument(9, this._strideX);
                    ForwardKernel.SetValueArgument(10, this._strideY);
                    ForwardKernel.SetValueArgument(11, this._padX);
                    ForwardKernel.SetValueArgument(12, this._padY);
                    ForwardKernel.SetValueArgument(13, this._kHeight);
                    ForwardKernel.SetValueArgument(14, this._kWidth);
                    ForwardKernel.SetValueArgument(15, this.OutputCount);
                    ForwardKernel.SetValueArgument(16, this.InputCount);

                    Weaver.CommandQueue.Execute
                    (
                        ForwardKernel,
                        null,
                        new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
                }
            }

            BatchArray output = BatchArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount);
            if (this._activation != null)
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray x)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (this._activation != null)
            {
                prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;
                this._prevOutput.RemoveAt(this._prevOutput.Count - 1);
            }

            Real[] gx = new Real[x.Data.Length];

            Real[] activatedgy = new Real[gy.BatchCount * gy.Length];
            for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
            {
                for (int och = 0; och < gy.Shape[0]; och++)
                {
                    for (int oy = 0; oy < gy.Shape[1]; oy++)
                    {
                        for (int ox = 0; ox < gy.Shape[2]; ox++)
                        {
                            int gyIndex = batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox;
                            Real gyData = gy.Data[gyIndex];
                            if (this._activation != null)
                            {
                                this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);
                            }
                            activatedgy[batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox] = gyData;

                            this.gb.Data[och] += gyData;
                        }
                    }
                }
            }

            if (!IsGpu)
            {
                for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
                {
                    for (int och = 0; och < gy.Shape[0]; och++)
                    {
                        //gWインデックス用
                        int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                        for (int oy = 0; oy < gy.Shape[1] * this._strideY; oy += this._strideY)
                        {
                            //計算省略のためにジャンプ
                            int kyStartIndex = this._padY - oy < 0 ? 0 : this._padY - oy;
                            int kyLimit = this._kHeight < x.Shape[1] - oy + this._padY ? this._kHeight : x.Shape[1] - oy + this._padY;

                            for (int ox = 0; ox < gy.Shape[2] * this._strideX; ox += this._strideX)
                            {
                                //計算省略のためにジャンプ
                                int kxStartIndex = this._padX - ox < 0 ? 0 : this._padX - ox;
                                int kxLimit = this._kWidth < x.Shape[2] - ox + this._padX ? this._kWidth : x.Shape[2] - ox + this._padX;

                                int gyIndex = batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox;

                                Real gyData = activatedgy[gyIndex]; //gyIndex = ch * ox * oy

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

                                            this.gW.Data[wIndex] += x.Data[inputIndex] * gyData;

                                            gx[inputIndex] += this.W.Data[wIndex] * gyData;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                //gyは共通で使用
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
                {
                    using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                    using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                    {
                        this.BackwardgWKernel.SetMemoryArgument(0, gpugY);
                        this.BackwardgWKernel.SetMemoryArgument(1, gpuX);
                        this.BackwardgWKernel.SetMemoryArgument(2, gpugW);
                        this.BackwardgWKernel.SetValueArgument(3, gy.BatchCount);
                        this.BackwardgWKernel.SetValueArgument(4, this.InputCount);
                        this.BackwardgWKernel.SetValueArgument(5, gy.Shape[0]);
                        this.BackwardgWKernel.SetValueArgument(6, gy.Shape[1]);
                        this.BackwardgWKernel.SetValueArgument(7, gy.Shape[2]);
                        this.BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                        this.BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                        this.BackwardgWKernel.SetValueArgument(10, x.Length);
                        this.BackwardgWKernel.SetValueArgument(11, this._strideX);
                        this.BackwardgWKernel.SetValueArgument(12, this._strideY);
                        this.BackwardgWKernel.SetValueArgument(13, this._padX);
                        this.BackwardgWKernel.SetValueArgument(14, this._padY);
                        this.BackwardgWKernel.SetValueArgument(15, this._kHeight);
                        this.BackwardgWKernel.SetValueArgument(16, this._kWidth);

                        Weaver.CommandQueue.Execute
                        (
                            this.BackwardgWKernel,
                            null,
                            new long[] { OutputCount * InputCount, this._kHeight, this._kWidth },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    }

                    using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                    using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                    {
                        this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                        this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                        this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                        this.BackwardgXKernel.SetValueArgument(3, this.OutputCount);
                        this.BackwardgXKernel.SetValueArgument(4, this.InputCount);
                        this.BackwardgXKernel.SetValueArgument(5, gy.Shape[0]);
                        this.BackwardgXKernel.SetValueArgument(6, gy.Shape[1]);
                        this.BackwardgXKernel.SetValueArgument(7, gy.Shape[2]);
                        this.BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                        this.BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                        this.BackwardgXKernel.SetValueArgument(10, x.Length);
                        this.BackwardgXKernel.SetValueArgument(11, this._strideX);
                        this.BackwardgXKernel.SetValueArgument(12, this._strideY);
                        this.BackwardgXKernel.SetValueArgument(13, this._padX);
                        this.BackwardgXKernel.SetValueArgument(14, this._padY);
                        this.BackwardgXKernel.SetValueArgument(15, this._kHeight);
                        this.BackwardgXKernel.SetValueArgument(16, this._kWidth);

                        Weaver.CommandQueue.Execute
                        (
                            this.BackwardgXKernel,
                            null,
                            new long[] { gy.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                    }
                }
            }

            return BatchArray.Convert(gx, x.Shape, x.BatchCount);
        }
    }
}
