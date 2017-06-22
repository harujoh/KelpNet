using System;
using System.Collections.Generic;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Activations;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousInputFunction
    {
        private readonly Activation _activation;
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly bool noBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Real[,] initialW = null, Real[] initialb = null, string name = "Linear", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputCount, outputCount)
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
                this.W.Data = initialW.Cast<Real>().ToArray();
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = new NdArray(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            this._activation = activation ?? new DummyActivation();
            if (IsGpu)
            {
                this.ForwardKernelSource = this._activation.ForwardActivateFunctionString + ForwardKernelSource;
                this.BackwardKernelSource = this._activation.BackwardActivateFunctionString + BackwardKernelSource;

                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "LinearForward");
                BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LinearBackward");
            }
        }

        public override string ForwardKernelSource { get; } =
@"
__kernel void LinearForward(
	__global const Real *gpuX,
	__global const Real *gpuW, 
	__global       Real *gpuY,
 	         const int OutputCount,
 	         const int InputCount)
{
	int i = get_global_id(0);
	int batchCount = get_global_id(1);

    gpuX += batchCount * InputCount;
    gpuW += i * InputCount;
    gpuY += i + batchCount * OutputCount;

    Real gpuYSum = *gpuY;

    for (int j = 0; j < InputCount; j++)
    {
        gpuYSum += gpuX[j] * gpuW[j];
    }
    
    *gpuY = gpuYSum;
    ForwardActivate(gpuY);
}";

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

                        this._activation.ForwardActivate(ref y[i + batchCount * this.OutputCount]);
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
                            new long[] { OutputCount, x.BatchCount},
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            BatchArray output = BatchArray.Convert(y, new[] { OutputCount }, x.BatchCount);
            if (!(this._activation is DummyActivation))
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        public override string BackwardKernelSource { get; } =
@"
__kernel void LinearBackward(
	__global const Real *gpugY,
	__global const Real *gpuY,
	__global const Real *gpuX,
	__global const Real *gpuW, 
	__global       Real *gpugW, 
	__global       Real *gpugX, 
	         const int BatchCount,
	         const int OutputCount,
	         const int InputCount)
{
    int j = get_global_id(0);

    gpugW += j;
    gpuW += j;

    gpugX += j;
    gpuX += j;

    for(int b = 0; b < BatchCount; b += InputCount)
    {
        for(int i = 0; i < OutputCount; i += InputCount)
        {
            Real gy = *gpugY;
            BackwardActivate(*gpuY, &gy);

            gpugW[i] += gpuX[b] * gy;
            gpugX[b] += gpuW[i] * gy;
            
            gpuY++;
            gpugY++;
        }
    }
}
";
        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (!(this._activation is DummyActivation))
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
                        this._activation.BackwardActivate(ref gyData, prevOutputData[i + batchCount * this.OutputCount]);

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
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, gy.Data))
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutputData))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevInput.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gxData))
                {
                    BackwardKernel.SetMemoryArgument(0, gpugY);
                    BackwardKernel.SetMemoryArgument(1, gpuY);
                    BackwardKernel.SetMemoryArgument(2, gpuX);
                    BackwardKernel.SetMemoryArgument(3, gpuW);
                    BackwardKernel.SetMemoryArgument(4, gpugW);
                    BackwardKernel.SetMemoryArgument(5, gpugX);
                    BackwardKernel.SetValueArgument(6, gy.BatchCount * this.InputCount);
                    BackwardKernel.SetValueArgument(7, this.OutputCount * this.InputCount);
                    BackwardKernel.SetValueArgument(8, this.InputCount);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
                            null,
                            new long[] { this.InputCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gxData, true, null);
                }

                for (int batchCount = 0; batchCount < gy.BatchCount * this.OutputCount; batchCount += this.OutputCount)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        Real gyData = gy.Data[batchCount + i];
                        this._activation.BackwardActivate(ref gyData, prevOutputData[batchCount + i]);
                        this.gb.Data[i] += gyData;
                    }
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
