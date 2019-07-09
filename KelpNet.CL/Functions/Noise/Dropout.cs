using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class Dropout : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Dropout";

        private readonly Real dropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        public bool IsParallel { get; set; }

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            this.SetParallel(gpuEnable);
        }

        public bool SetParallel(bool enable)
        {
            this.IsParallel = enable & Weaver.Enable;

            if (IsParallel)
            {
                InitParallel();

                SingleInputForward = ForwardGpu;
                SingleOutputBackward = BackwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
                SingleOutputBackward = BackwardCpu;
            }

            return IsParallel;
        }

        public void InitParallel()
        {
            if (IsParallel)
            {
                string kernelSource = Weaver.GetKernelSource(Resources.Dropout);
                ComputeProgram program = Weaver.CreateProgram(kernelSource);

                ForwardKernel = program.CreateKernel("DropoutForward");
                BackwardKernel = program.CreateKernel("DropoutBackward");
            }
        }

        private Real[] MakeMask(int xLength)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            this.maskStack.Add(mask);

            return mask;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public NdArray ForwardGpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuMask);
                ForwardKernel.SetMemoryArgument(2, gpuY);
                ForwardKernel.SetValueArgument(3, mask.Length);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { x.Data.Length },
                    null,
                    null
                );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public void BackwardCpu(NdArray y, NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * y.Length + i] *= mask[i];
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }

        public void BackwardGpu(NdArray y, NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, result))
            {
                BackwardKernel.SetMemoryArgument(0, gpuMask);
                BackwardKernel.SetMemoryArgument(1, gpugX);
                BackwardKernel.SetValueArgument(2, y.Length);

                Weaver.CommandQueue.Execute
                (
                    BackwardKernel,
                    null,
                    new long[] { mask.Length, y.BatchCount },
                    null,
                    null
                );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpugX, ref result, true, null);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }


        //Predict時に何もしない
        public override NdArray Predict(NdArray input)
        {
            return input;
        }
    }
}
