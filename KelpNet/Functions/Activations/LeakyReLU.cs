using System;
using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : Activation
    {
        private readonly Real _slope;

        public override string ForwardActivateGPU { get; }
        public override string BackwardActivateGPU { get; }

        public override string ForwardKernelSource { get; }
        public override string BackwardKernelSource { get; }

        public LeakyReLU(Real? slope = null, string name = "LeakyReLU", bool isGpu = true) : base(name, isGpu)
        {
            this._slope = slope ?? (Real)0.2;

            if (IsGpu)
            {
                ForwardActivateGPU = String.Format(
                    @"
void ForwardActivate(__global Real* gpuY)
{{
    if(*gpuY < 0.0)
    {{
        *gpuY *= {0};
    }}
}}
", this._slope);

                BackwardActivateGPU = String.Format(
                    @"
void BackwardActivate(Real gpuY, __global Real* gpugX)
{{
    if(gpuY <= 0.0)
    {{
        *gpugX = gpuY * {0};
    }}
}}
", this._slope);

                this.ForwardKernelSource = ForwardActivateGPU + String.Format(ForwardKernelString, this.ForwardKernelName);
                this.BackwardKernelSource = BackwardActivateGPU + String.Format(BackwardKernelString, this.BackwardKernelName);

                this.ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, this.ForwardKernelName);
                this.BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, this.BackwardKernelName);
            }
        }

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x *= this._slope;
            }
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = y * this._slope;
            }
        }

        //        public override void InitKernel()
        //        {
        //            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "LeakyReLUForward");
        //            BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LeakyReLUBackward");
        //        }

        //        const string ForwardKernelSource =
        //@"
        //__kernel void LeakyReLUForward(
        //	         const Real slope,
        //	__global Real* gpuY)
        //{
        //	int i = get_global_id(0);

        //    if(gpuY[i] < 0.0)
        //    {
        //        gpuY[i] *= slope;
        //    }
        //}";

        //        protected override BatchArray NeedPreviousForward(BatchArray x)
        //        {
        //            Real[] y = x.Data.ToArray();

        //            if (!IsGpu)
        //            {
        //                for (int i = 0; i < x.Data.Length; i++)
        //                {
        //                    if (y[i] < 0)
        //                    {
        //                        y[i] *= this._slope;
        //                    }
        //                }
        //            }
        //            else
        //            {
        //                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
        //                {
        //                    ForwardKernel.SetValueArgument(0, this._slope);
        //                    ForwardKernel.SetMemoryArgument(1, gpuY);

        //                    Weaver.CommandQueue.Execute
        //                        (
        //                            ForwardKernel,
        //                            null,
        //                            new long[] { x.Data.Length },
        //                            null,
        //                            null
        //                        );

        //                    Weaver.CommandQueue.Finish();
        //                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
        //                }
        //            }

        //            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        //        }

        //        const string BackwardKernelSource =
        //@"
        //#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
        //#pragma OPENCL EXTENSION cl_khr_fp64 : enable
        //#endif

        //__kernel void LeakyReLUBackward(
        //	__global Real *gpuY,
        //  	   const Real slope,
        //	__global Real *gpugX)
        //{
        //	int i = get_global_id(0);

        //    if(gpuY[i] <= 0.0)
        //    {
        //        gpugX[i] = gpuY[i] * slope;
        //    }
        //}";

        //        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        //        {
        //            Real[] gx = gy.Data.ToArray();

        //            if (!IsGpu)
        //            {
        //                for (int i = 0; i < gy.Data.Length; i++)
        //                {
        //                    if (prevOutput.Data[i] <= 0)
        //                    {
        //                        gx[i] = prevOutput.Data[i] * this._slope;
        //                    }
        //                }
        //            }
        //            else
        //            {
        //                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutput.Data))
        //                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
        //                {
        //                    BackwardKernel.SetMemoryArgument(0, gpuY);
        //                    BackwardKernel.SetValueArgument(1, this._slope);
        //                    BackwardKernel.SetMemoryArgument(2, gpugX);

        //                    Weaver.CommandQueue.Execute
        //                        (
        //                            BackwardKernel,
        //                            null,
        //                            new long[] { gy.Data.Length },
        //                            null,
        //                            null
        //                        );

        //                    Weaver.CommandQueue.Finish();
        //                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
        //                }
        //            }

        //            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        //        }
    }
}
