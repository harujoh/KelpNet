using System;
using System.Collections.Generic;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : CompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

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

        private readonly bool noBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool isGpu = false, CompressibleActivation activation = null) : base(name, inputCount, outputCount)
        {
            this.noBias = noBias;
            this.W = new NdArray(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                this.W.Data = Real.GetArray(initialW);
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = new NdArray(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    this.b.Data = Real.GetArray(initialb);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            this.Activation = activation;

            if (isGpu)
            {
                InitGpu();
            }
        }

        protected override void CreateKernel()
        {
            string kernelSource = Weaver.GetKernelSource(FUNCTION_NAME);

            if (this.Activation != null)
            {
                kernelSource = this.Activation.ActivateFunctionString + kernelSource.Replace("/*ForwardActivate*/", "ForwardActivate(gpuY);");
            }

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            ForwardKernel = program.CreateKernel("LinearForward");
            BackwardgWKernel = program.CreateKernel("LineargWBackward");
            BackwardgXKernel = program.CreateKernel("LineargXBackward");
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            Real[] y = new Real[OutputCount * x.BatchCount];

            if (!IsGpu)
            {
                for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        y[i + batchCount * this.OutputCount] += this.b.Data[i];

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            y[i + batchCount * this.OutputCount] += x.Data[j + batchCount * this.InputCount] * this.W.Data[i * this.InputCount + j];
                        }

                        if (this.Activation != null) this.Activation.ForwardActivate(ref y[i + batchCount * this.OutputCount]);
                    }
                }
            }
            else
            {
                if (!this.noBias)
                {
                    for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
                    {
                        Array.Copy(this.b.Data, 0, y, batchCount * this.OutputCount, this.b.Data.Length);
                    }
                }

                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
                {
                    ForwardKernel.SetMemoryArgument(0, gpuX);
                    ForwardKernel.SetMemoryArgument(1, gpuW);
                    ForwardKernel.SetMemoryArgument(2, gpuY);
                    ForwardKernel.SetValueArgument(3, this.OutputCount);
                    ForwardKernel.SetValueArgument(4, this.InputCount);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
                            null,
                            new long[] { OutputCount, x.BatchCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            BatchArray output = BatchArray.Convert(y, new[] { OutputCount }, x.BatchCount);
            if (this.Activation != null)
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (this.Activation != null)
            {
                prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;
                this._prevOutput.RemoveAt(this._prevOutput.Count - 1);
            }

            Real[] gxData = new Real[prevInput.Data.Length];

            if (!IsGpu)
            {
                for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        Real gyData = gy.Data[i + batchCount * this.OutputCount];
                        if (this.Activation != null) this.Activation.BackwardActivate(ref gyData, prevOutputData[i + batchCount * this.OutputCount]);

                        this.gb.Data[i] += gyData;

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            this.gW.Data[i * this.InputCount + j] += prevInput.Data[j + batchCount * this.InputCount] * gyData;
                            gxData[j + batchCount * this.InputCount] += this.W.Data[i * this.InputCount + j] * gyData;
                        }
                    }
                }
            }
            else
            {
                Real[] activatedgY = new Real[gy.Data.Length];

                for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        Real gyData = gy.Data[i + batchCount * this.OutputCount];
                        if (this.Activation != null) this.Activation.BackwardActivate(ref gyData, prevOutputData[i + batchCount * this.OutputCount]);
                        activatedgY[i + batchCount * this.OutputCount] = gyData;

                        this.gb.Data[i] += gyData;
                    }
                }

                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgY))
                {
                    using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                    using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevInput.Data))
                    {
                        BackwardgWKernel.SetMemoryArgument(0, gpugY);
                        BackwardgWKernel.SetMemoryArgument(1, gpuX);
                        BackwardgWKernel.SetMemoryArgument(2, gpugW);
                        BackwardgWKernel.SetValueArgument(3, gy.BatchCount);
                        BackwardgWKernel.SetValueArgument(4, this.OutputCount);
                        BackwardgWKernel.SetValueArgument(5, this.InputCount);

                        Weaver.CommandQueue.Execute
                        (
                            BackwardgWKernel,
                            null,
                            new long[] { this.InputCount, this.OutputCount },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    }

                    using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gxData.Length))
                    using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                    {
                        BackwardgXKernel.SetMemoryArgument(0, gpugY);
                        BackwardgXKernel.SetMemoryArgument(1, gpuW);
                        BackwardgXKernel.SetMemoryArgument(2, gpugX);
                        BackwardgXKernel.SetValueArgument(3, gy.BatchCount);
                        BackwardgXKernel.SetValueArgument(4, this.OutputCount);
                        BackwardgXKernel.SetValueArgument(5, this.InputCount);

                        Weaver.CommandQueue.Execute
                        (
                            BackwardgXKernel,
                            null,
                            new long[] { this.InputCount, gy.BatchCount },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gxData, true, null);
                    }
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
