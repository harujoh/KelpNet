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
        public bool NoBias { get; set; }


        [DataMember]
        public int InputCount { get; set; }

        [DataMember]
        public int OutputCount { get; set; }


        [DataMember]
        public ICompressibleActivation<T> Activation { get; set; }

        public Linear(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.OutputCount = outputCount;
            this.InputCount = inputCount;

            this.Weight = new NdArray<T>(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.Activation = activation;

            if (initialW == null)
            {
                Initializer.InitHeNorm(this.Weight);
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

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Linear<float> linearF:
                    linearF.SingleInputForward = x => LinearF.SingleInputForward(x, linearF.Weight.Data, linearF.Bias, linearF.NoBias, linearF.OutputCount, linearF.InputCount, linearF.Activation, linearF);
                    linearF.SingleOutputBackward = (y, x) => LinearF.SingleOutputBackward(y, x, linearF.Weight, linearF.Bias, linearF.NoBias, linearF.OutputCount, linearF.InputCount, linearF.Activation);
                    break;

                case Linear<double> linearD:
                    linearD.SingleInputForward = x => LinearD.SingleInputForward(x, linearD.Weight.Data, linearD.Bias, linearD.NoBias, linearD.OutputCount, linearD.InputCount, linearD.Activation, linearD);
                    linearD.SingleOutputBackward = (y, x) => LinearD.SingleOutputBackward(y, x, linearD.Weight, linearD.Bias, linearD.NoBias, linearD.OutputCount, linearD.InputCount, linearD.Activation);
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
    public static class LinearD
#else
    public static class LinearF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real[] Weight, NdArray<Real> Bias, bool NoBias, int OutputCount, int InputCount, ICompressibleActivation<Real> Activation, IFunction<Real> linear)
        {
            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount, OutputCount, Bias.Data);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    for (int j = 0; j < InputCount; j++)
                    {
                        y[batchCount * OutputCount + i] += x.Data[batchCount * InputCount + j] * Weight[i * InputCount + j];
                    }
                }
            }

            if (Activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = Activation.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, linear);
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

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> Weight, NdArray<Real> bias, bool NoBias, int OutputCount, int InputCount, ICompressibleActivation<Real> activation)
        {
            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount, OutputCount, bias.Grad);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * OutputCount];

                    for (int j = 0; j < InputCount; j++)
                    {
                        Weight.Grad[i * InputCount + j] += x.Data[batchCount * InputCount + j] * gyData;
                        x.Grad[batchCount * InputCount + j] += Weight.Data[i * InputCount + j] * gyData;
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
