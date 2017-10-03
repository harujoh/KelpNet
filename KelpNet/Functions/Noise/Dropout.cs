using System;
using System.Collections.Generic;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : Function, IParallelizable
    {
        const string FUNCTION_NAME = "Dropout";

        private readonly Real dropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardKernel;

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, bool gpuEnable = false) : base(name)
        {
            this.dropoutRatio = dropoutRatio;

            this.SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            if (GpuEnable)
            {
                CreateKernel();
                Forward = ForwardGpu;
                Backward = BackwardGpu;
            }
            else
            {
                Forward = ForwardCpu;
                Backward = BackwardCpu;
            }

            return GpuEnable;
        }

        public void CreateKernel()
        {
            string kernelSource = Weaver.GetKernelSource(FUNCTION_NAME);
            ComputeProgram program = Weaver.CreateProgram(kernelSource);

            ForwardKernel = program.CreateKernel("DropoutForward");
            BackwardKernel = program.CreateKernel("DropoutBackward");
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

        public BatchArray ForwardCpu(BatchArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        public BatchArray ForwardGpu(BatchArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
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

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        public BatchArray BackwardCpu(BatchArray gy)
        {
            Real[] result = gy.Data.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            for (int b = 0; b < gy.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * gy.Length + i] *= mask[i];
                }
            }

            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }

        public BatchArray BackwardGpu(BatchArray gy)
        {
            Real[] result = gy.Data.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, result))
            {
                BackwardKernel.SetMemoryArgument(0, gpuMask);
                BackwardKernel.SetMemoryArgument(1, gpugX);
                BackwardKernel.SetValueArgument(2, gy.Length);

                Weaver.CommandQueue.Execute
                (
                    BackwardKernel,
                    null,
                    new long[] { mask.Length, gy.BatchCount },
                    null,
                    null
                );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpugX, ref result, true, null);
            }

            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }


        //Predict時に何もしない
        public override BatchArray Predict(BatchArray input)
        {
            return input;
        }
    }
}
