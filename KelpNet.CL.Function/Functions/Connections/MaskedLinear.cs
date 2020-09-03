using System;
using System.Runtime.Serialization;
using KelpNet.CL.Common;

#if DOUBLE
using Real = System.Double;
#else
using KelpNet.CL.Properties;
using Real = System.Single;
#endif

namespace KelpNet.CL
{
#if !DOUBLE
    [DataContract(Name = "MaskedLinear", Namespace = "KelpNet")]
    public class MaskedLinear<T> : Linear<T> where T : unmanaged, IComparable<T>
    {
        [DataMember]
        public NdArray<T> Mask { get; set; }

        public MaskedLinear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, T[] initialb = null, Action<NdArray<T>> weightInitializer = null, ICompressibleActivation<T> activation = null, string name = "Linear", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(inputCount, outputCount, noBias, initialW, initialb, weightInitializer, activation, name, inputNames, outputNames, gpuEnable)
        {
            this.Mask = new NdArray<T>(outputCount, inputCount);
            this.Mask.InitGrad();//Maskは更新されない非パラメータなので自分で初期化する
            this.InitMaskedFunc(new StreamingContext());
        }

        public MaskedLinear(CPU.MaskedLinear<T> maskedLinear) : base(maskedLinear)
        {
            this.Mask = maskedLinear.Mask;
            this.InitMaskedFunc(new StreamingContext());
        }

        [OnDeserializing]
        protected void InitMaskedFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case MaskedLinear<float> linearF:
                        linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.ForwardKernel, linearF);
                        linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.BackwardgWKernel, linearF.BackwardgXKernel, linearF.Activation);
                        break;

                    case MaskedLinear<double> linearD:
                        linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.ForwardKernel, linearD);
                        linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.BackwardgWKernel, linearD.BackwardgXKernel, linearD.Activation);
                        break;
                }
            }
            else
            {
                switch (this)
                {
                    case MaskedLinear<float> linearF:
                        linearF.SingleInputForward = x => CPU.LinearF.SingleInputForward(x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.Activation, linearF);
                        linearF.SingleOutputBackward = (y, x) => CPU.LinearF.SingleOutputBackward(y, x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.Activation);
                        break;

                    case MaskedLinear<double> linearD:
                        linearD.SingleInputForward = x => CPU.LinearD.SingleInputForward(x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.Activation, linearD);
                        linearD.SingleOutputBackward = (y, x) => CPU.LinearD.SingleOutputBackward(y, x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.Activation);
                        break;
                }

            }
        }

        protected override void DefaultInitWeight()
        {
            Initializer.InitXavier(this.Weight);
        }

        public override CPU.Convolution2D<T> AsConvolution2D()
        {
            return new Convolution2D<T>(this);
        }
    }
#endif

#if DOUBLE
    public static partial class LinearD
#else
    public static partial class LinearF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> mask, NdArray<Real> weight, NdArray<Real> bias, ComputeKernel forwardKernel, IFunction<Real> linear)
        {
            return SingleInputForward(x, weight * mask, bias, forwardKernel, linear);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> mask, NdArray<Real> weight, NdArray<Real> bias, ComputeKernel backwardgWKernel, ComputeKernel backwardgXKernel, ICompressibleActivation<Real> activation)
        {
            NdArray<Real> maskedWeight = weight * mask;
            maskedWeight.InitGrad();//MaskedWeightはOptimizerの対象にならない非パラメータの為初期化が必要

            SingleOutputBackward(y, x, maskedWeight, bias, backwardgWKernel, backwardgXKernel, activation);

            for (int i = 0; i < weight.Data.Length; i++)
            {
                mask.Grad[i] = maskedWeight.Grad[i];//マスク前の重みの傾きをマスクの傾きへ退避
                weight.Grad[i] += mask.Data[i] * maskedWeight.Grad[i];//マスクした傾きを適用
            }
        }
    }
}
