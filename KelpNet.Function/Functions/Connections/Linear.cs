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
    [DataContract(Name = "Linear", Namespace = "KelpNet")]
    public class Linear<T> : SingleInputFunction<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Linear";

        [DataMember]
        public NdArray<T> Weight { get; set; }

        [DataMember]
        public NdArray<T> Bias { get; set; }


        [DataMember]
        public ICompressibleActivation<T> Activation { get; set; }

        public Linear(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, Action<NdArray<T>> weightInitializer = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Weight = new NdArray<T>(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.Activation = activation;

            if (initialW == null)
            {
                if (weightInitializer == null)
                {
                    DefaultInitWeight();
                }
                else
                {
                    weightInitializer(this.Weight);
                }
            }
            else
            {
                this.Weight.Data = initialW.FlattenEx<T>();
            }

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

        protected virtual void DefaultInitWeight()
        {
            Initializer.InitHeNorm(this.Weight);
        }

        [OnDeserializing]
        protected void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Linear<float> linearF:
                    linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Weight, linearF.Bias, linearF.Activation, linearF);
                    linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Weight, linearF.Bias, linearF.Activation);
                    break;

                case Linear<double> linearD:
                    linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Weight, linearD.Bias, linearD.Activation, linearD);
                    linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Weight, linearD.Bias, linearD.Activation);
                    break;
            }
        }

        public virtual Convolution2D<T> AsConvolution2D()
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
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, ICompressibleActivation<Real> activation, IFunction<Real> linear)
        {
            int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];

            Real[] y = bias == null ? new Real[outputCount * x.BatchCount] : GetBiasedValue(x.BatchCount, outputCount, bias.Data);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < outputCount; i++)
                {
                    for (int j = 0; j < inputCount; j++)
                    {
                        y[batchCount * outputCount + i] += x.Data[batchCount * inputCount + j] * weight.Data[i * inputCount + j];
                    }
                }
            }

            if (activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = activation.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { outputCount }, x.BatchCount, linear);
        }

        public static Real[] GetBiasedValue(int batchCount, int outputCount, Real[] bias)
        {
            Real[] y = new Real[outputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(bias, 0, y, i * outputCount, bias.Length);
            }

            return y;
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, ICompressibleActivation<Real> activation)
        {
            int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];

            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y, x) : y.Grad;
            if (bias != null) CalcBiasGrad(activatedgy, y.BatchCount, outputCount, bias.Grad);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < outputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * outputCount];

                    for (int j = 0; j < inputCount; j++)
                    {
                        weight.Grad[i * inputCount + j] += x.Data[batchCount * inputCount + j] * gyData;
                        x.Grad[batchCount * inputCount + j] += weight.Data[i * inputCount + j] * gyData;
                    }
                }
            }
        }

        public static void CalcBiasGrad(Real[] gy, int batchCount, int OutputCount, Real[] biasGrad)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    biasGrad[i] += gy[batchCounter * OutputCount + i];
                }
            }
        }
    }
}
