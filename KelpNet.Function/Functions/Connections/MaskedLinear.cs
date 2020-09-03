using System;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "MaskedLinear", Namespace = "KelpNet")]
    public class MaskedLinear<T> : Linear<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "MaskedLinear";

        [DataMember]
        public NdArray<T> Mask { get; set; }

        public MaskedLinear(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitMaskFunc(new StreamingContext());
        }

        public MaskedLinear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, Action<NdArray<T>> weightInitializer = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(inputCount, outputCount, noBias, initialW, initialb, weightInitializer, activation, name, inputNames, outputNames)
        {
            this.Mask = new NdArray<T>(outputCount, inputCount);
            this.Mask.InitGrad();//Maskは更新されない非パラメータなので自分で初期化する
            InitMaskFunc(new StreamingContext());
        }

        protected override void DefaultInitWeight()
        {
            Initializer.InitXavier(this.Weight);
        }

        [OnDeserializing]
        protected void InitMaskFunc(StreamingContext sc)
        {
            switch (this)
            {
                case MaskedLinear<float> linearF:
                    linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.Activation, linearF);
                    linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Mask, linearF.Weight, linearF.Bias, linearF.Activation);
                    break;

                case MaskedLinear<double> linearD:
                    linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.Activation, linearD);
                    linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Mask, linearD.Weight, linearD.Bias, linearD.Activation);
                    break;
            }
        }

        //public virtual MaskedConvolution2D<T> AsMakedConvolution2D()
        //{
        //    return new MaskedConvolution2D<T>(this);
        //}
    }
#endif

#if DOUBLE
    public static partial class LinearD
#else
    public static partial class LinearF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> mask, NdArray<Real> weight, NdArray<Real> bias, ICompressibleActivation<Real> activation, IFunction<Real> linear)
        {
            return SingleInputForward(x, weight * mask, bias, activation, linear);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> mask, NdArray<Real> weight, NdArray<Real> bias, ICompressibleActivation<Real> activation)
        {
            NdArray<Real> maskedWeight = weight * mask;
            maskedWeight.InitGrad();//MaskedWeightはOptimizerの対象にならない非パラメータの為初期化が必要

            SingleOutputBackward(y, x, maskedWeight, bias, activation);

            for (int i = 0; i < weight.Data.Length; i++)
            {
                mask.Grad[i] = maskedWeight.Grad[i];//マスク前の重みの傾きをマスクの傾きへ退避
                weight.Grad[i] += mask.Data[i] * maskedWeight.Grad[i];//マスクした傾きを適用
            }
        }
    }
}
