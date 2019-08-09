using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "Dropout", Namespace = "KelpNet")]
    public class Dropout : CPU.Dropout, IParallelizable
    {
        public string FunctionName => "Dropout";
        public string KernelSource => OpenCL.GetKernelSource(Resources.Dropout);

        //[NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        //[NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        [DataMember]
        public bool IsParallel { get; set; }

        public Dropout(double dropoutRatio = 0.5, string name = "Dropout", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(dropoutRatio, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
        }

        public Dropout(CPU.Dropout dropout) : base(dropout.Name, dropout.InputNames, dropout.OutputNames)
        {
            this.SetParallel(true);
        }

        public bool SetParallel(bool enable)
        {
            this.IsParallel = enable & OpenCL.Enable;

            if (IsParallel)
            {
                ComputeProgram program = OpenCL.CreateProgram(this.KernelSource);

                ForwardKernel = program.CreateKernel("DropoutForward");
                BackwardKernel = program.CreateKernel("DropoutBackward");
            }

            return IsParallel;
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            //フラグチェック
            if (!IsParallel) return base.SingleInputForward(x);

            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuMask);
                ForwardKernel.SetMemoryArgument(2, gpuY);
                ForwardKernel.SetValueArgument(3, mask.Length);

                OpenCL.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { x.Data.Length },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            if (!IsParallel)
            {
                base.SingleOutputBackward(y, x);
                return;
            }

            Real[] result = y.Grad.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, mask))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, result))
            {
                BackwardKernel.SetMemoryArgument(0, gpuMask);
                BackwardKernel.SetMemoryArgument(1, gpugX);
                BackwardKernel.SetValueArgument(2, y.Length);

                OpenCL.CommandQueue.Execute
                (
                    BackwardKernel,
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

        //Predict時に何もしない
        public override NdArray Predict(NdArray input)
        {
            return input;
        }
    }
}
