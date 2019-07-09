using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public abstract class CompressibleActivation : KelpNet.CompressibleActivation
    {
        const string FUNCTION_NAME = "Activation";

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        //GPU向けのActivate関数の文字列
        public string ActivateFunctionString; //外部の関数に組み込まれる
        protected string ActivateKernelString; //単品で呼ぶ用

        public readonly KeyValuePair<string, string>[] ActivationParameters;

        public string ForwardKernelName { get; }
        public string BackwardKernelName { get; }

        protected CompressibleActivation(string functionName, string activateFunctionString, KeyValuePair<string, string>[] parameters, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");

            this.ActivateFunctionString = activateFunctionString;

            this.ActivateKernelString = OpenCL.GetKernelSource(Resources.Activation).Replace("/*kernelNameBase*/", kernelNameBase);

            if (parameters == null)
            {
                ActivationParameters = new KeyValuePair<string, string>[] { };
            }
            else
            {
                ActivationParameters = parameters;
            }

            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardKernelName = kernelNameBase + "Backward";

            this.SetParallel(gpuEnable);
        }

        public override bool SetParallel(bool enable)
        {
            this.IsParallel = enable & OpenCL.Enable;

            if (this.IsParallel)
            {
                this.InitParallel();

                this.SingleInputForward = this.NeedPreviousForwardGpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardGpu;
            }
            else
            {
                this.SingleInputForward = this.NeedPreviousForwardCpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
            }

            return IsParallel;
        }

        public override void InitParallel()
        {
            if (this.IsParallel)
            {
                string kernelSource = this.ActivateFunctionString;

                foreach (var parameter in this.ActivationParameters)
                {
                    kernelSource = this.ActivateFunctionString.Replace(parameter.Key, parameter.Value);
                }

                kernelSource += this.ActivateKernelString;

                ComputeProgram program = OpenCL.CreateProgram(kernelSource);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
            }
        }

        private NdArray NeedPreviousForwardGpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, y.Length))
            {
                this.ForwardKernel.SetMemoryArgument(0, gpuX);
                this.ForwardKernel.SetMemoryArgument(1, gpuY);

                OpenCL.CommandQueue.Execute
                    (
                        this.ForwardKernel,
                        null,
                        new long[] { x.Data.Length },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, y.Grad))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, y.Data))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
            {
                this.BackwardKernel.SetMemoryArgument(0, gpugY);
                this.BackwardKernel.SetMemoryArgument(1, gpuY);
                this.BackwardKernel.SetMemoryArgument(2, gpugX);

                OpenCL.CommandQueue.Execute
                    (
                        this.BackwardKernel,
                        null,
                        new long[] { y.Grad.Length },
                        null,
                        null
                    );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
