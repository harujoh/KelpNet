using System;
using System.Runtime.Serialization;
using KelpNet.CPU;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [DataContract(Name = "MaskedLinear", Namespace = "KelpNet")]
    public class MaskedLinear<T> : SingleInputFunction<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "MaskedLinear";

        [DataMember]
        public NdArray<T> Weight { get; set; }

        [DataMember]
        public NdArray<T> Mask { get; set; }

        [DataMember]
        public NdArray<T> Bias { get; set; }


        [DataMember]
        public ICompressibleActivation<T> Activation { get; set; }

        public MaskedLinear(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public MaskedLinear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Weight = new NdArray<T>(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.Activation = activation;

            if (initialW == null)
            {
                Initializer.InitXavier(this.Weight);
            }
            else
            {
                this.Weight.Data = initialW.FlattenEx<T>();
            }

            this.Mask = new NdArray<T>(Weight.Length);

            this.Parameters[0] = this.Weight;

            if (!noBias)
            {
                this.Bias = new NdArray<T>(outputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = initialb.FlattenEx<T>();
                }

                this.Parameters[1] = this.Bias;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case MaskedLinear<float> linearF:
                    linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Weight * linearF.Mask, linearF.Bias, linearF.Activation, linearF);
                    linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Weight, linearF.Bias, linearF.Activation);
                    break;

                case MaskedLinear<double> linearD:
                    linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Weight * linearD.Mask, linearD.Bias, linearD.Activation, linearD);
                    linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Weight, linearD.Bias, linearD.Activation);
                    break;
            }
        }

        //関数の中身は通常のLinearをそのまま使用する
    }
#endif
}
