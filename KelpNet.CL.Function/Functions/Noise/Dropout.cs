using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
using KelpNet.CL.Properties;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "Dropout", Namespace = "KelpNet")]
    public class Dropout<T> : CPU.Dropout<T>, IParallelizable where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Dropout";
        public string KernelSource { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        [DataMember]
        public bool IsParallel { get; set; }

        public Dropout(T? dropoutRatio = null, string name = "Dropout", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(dropoutRatio, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Dropout(CPU.Dropout<T> dropout) : base(dropout.Name, dropout.InputNames, dropout.OutputNames)
        {
            this.DropoutRatio = dropout.DropoutRatio;

            this.SetParallel(true);
        }

        public bool SetParallel(bool enable)
        {
            KernelSource = OpenCL.GetKernelSource(Resources.Dropout);
            this.IsParallel = enable & OpenCL.Enable;

            if (IsParallel)
            {
                ComputeProgram program = OpenCL.CreateProgram<T>(this.KernelSource);

                ForwardKernel = program.CreateKernel("DropoutForward");
                BackwardKernel = program.CreateKernel("DropoutBackward");
            }

            this.InitFunc(new StreamingContext());

            return IsParallel;
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case Dropout<float> dropOutF:
                        dropOutF.SingleInputForward = x => DropOutF.SingleInputForward(x, CPU.DropOutF.MakeMask, dropOutF.DropoutRatio, dropOutF.maskStack, dropOutF.ForwardKernel, dropOutF);
                        dropOutF.SingleOutputBackward = (y, x) => DropOutF.SingleOutputBackward(y, x, dropOutF.maskStack, dropOutF.BackwardKernel);
                        break;

                    case Dropout<double> dropOutD:
                        dropOutD.SingleInputForward = x => DropOutD.SingleInputForward(x, CPU.DropOutD.MakeMask, dropOutD.DropoutRatio, dropOutD.maskStack, dropOutD.ForwardKernel, dropOutD);
                        dropOutD.SingleOutputBackward = (y, x) => DropOutD.SingleOutputBackward(y, x, dropOutD.maskStack, dropOutD.BackwardKernel);
                        break;
                }
            }
        }
    }
#endif

#if DOUBLE
    public static class DropOutD
#else
    public static class DropOutF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Func<int, Real, List<Real[]>, Real[]> makeMask, Real dropoutRatio, List<Real[]> maskStack, ComputeKernel forwardKernel, IFunction<Real> dropout)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = makeMask(x.Length, dropoutRatio, maskStack);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                forwardKernel.SetMemoryArgument(0, gpuX);
                forwardKernel.SetMemoryArgument(1, gpuMask);
                forwardKernel.SetMemoryArgument(2, gpuY);
                forwardKernel.SetValueArgument(3, mask.Length);

                OpenCL.CommandQueue.Execute
                (
                    forwardKernel,
                    null,
                    new long[] { x.Data.Length },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, dropout);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, List<Real[]> maskStack, ComputeKernel backwardKernel)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, result))
            {
                backwardKernel.SetMemoryArgument(0, gpuMask);
                backwardKernel.SetMemoryArgument(1, gpugX);
                backwardKernel.SetValueArgument(2, y.Length);

                OpenCL.CommandQueue.Execute
                (
                    backwardKernel,
                    null,
                    new long[] { mask.Length, y.BatchCount },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpugX, ref result, true, null);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }
    }
}
