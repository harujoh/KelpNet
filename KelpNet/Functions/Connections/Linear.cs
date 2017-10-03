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

        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "ForwardActivate(gpuY);";

        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public readonly bool NoBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool gpuEnable = false, CompressibleActivation activation = null) : base(name, inputCount, outputCount, gpuEnable, FUNCTION_NAME, activation, new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE))
        {
            this.W = new NdArray(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);
            this.NoBias = noBias;

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

            if (!noBias)
            {
                this.gb = NdArray.ZerosLike(this.b);

                if (initialb != null)
                {
                    this.b.Data = Real.GetArray(initialb);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(this.b.Data, 0, y, i * this.OutputCount, this.b.Data.Length);
            }

            return y;
        }

        protected override BatchArray NeedPreviousForwardCpu(BatchArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    for (int j = 0; j < this.InputCount; j++)
                    {
                        y[batchCount * this.OutputCount + i] += x.Data[batchCount * this.InputCount + j] * this.W.Data[i * this.InputCount + j];
                    }
                }
            }

            if (this.Activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    this.Activation.ForwardActivate(ref y[i]);
                }
            }

            return GetForwardResult(y, x.BatchCount);
        }

        protected override BatchArray NeedPreviousForwardGpu(BatchArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

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

            return GetForwardResult(y, x.BatchCount);
        }

        BatchArray GetForwardResult(Real[] y, int batchCount)
        {
            BatchArray result = BatchArray.Convert(y, new[] { OutputCount }, batchCount);

            if (this.Activation != null)
            {
                this._prevOutput.Add(result);
            }

            return result;
        }

        Real[] GetActivatedgy(BatchArray gy)
        {
            Real[] activatedgY = new Real[gy.Data.Length];
            var prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;

            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real gyData = gy.Data[i + batchCount * this.OutputCount];

                    this.Activation.BackwardActivate(ref gyData, prevOutputData[i + batchCount * this.OutputCount]);
                    activatedgY[i + batchCount * this.OutputCount] = gyData;
                }
            }

            return activatedgY;
        }

        void CalcBiasGrad(Real[] gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    this.gb.Data[i] += gy[batchCounter * this.OutputCount + i];
                }
            }
        }

        protected override BatchArray NeedPreviousBackwardCpu(BatchArray gy, BatchArray prevInput)
        {
            Real[] gxData = new Real[prevInput.Data.Length];
            Real[] activatedgy = this.Activation != null ? GetActivatedgy(gy) : gy.Data;
            if (!NoBias) CalcBiasGrad(activatedgy, gy.BatchCount);

            for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * this.OutputCount];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.gW.Data[i * this.InputCount + j] += prevInput.Data[j + batchCount * this.InputCount] * gyData;
                        gxData[j + batchCount * this.InputCount] += this.W.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }

        protected override BatchArray NeedPreviousBackwardGpu(BatchArray gy, BatchArray prevInput)
        {
            Real[] gxData = new Real[prevInput.Data.Length];
            Real[] activatedgy = this.Activation != null ? GetActivatedgy(gy) : gy.Data;
            if (!NoBias) CalcBiasGrad(activatedgy, gy.BatchCount);

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
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

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
