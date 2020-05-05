using System;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class EmbedID<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "EmbedID";

        public NdArray<T> Weight;

        public EmbedID(int inputCount, int outputCount, Array initialW = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Weight = new NdArray<T>(inputCount, outputCount);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitHeNorm(this.Weight);
            }
            else
            {
                this.Weight.Data = initialW.FlattenEx<T>();
            }

            this.Parameters = new[] { this.Weight };

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case EmbedID<float> embedIdF:
                    embedIdF.SingleInputForward = (x) => EmbedIDF.SingleInputForward(x, embedIdF.Weight, embedIdF);
                    embedIdF.SingleOutputBackward = (y, x) => EmbedIDF.SingleOutputBackward(y, x, embedIdF.Weight);
                    break;

                case EmbedID<double> embedIdD:
                    embedIdD.SingleInputForward = (x) => EmbedIDD.SingleInputForward(x, embedIdD.Weight, embedIdD);
                    embedIdD.SingleOutputBackward = (y, x) => EmbedIDD.SingleOutputBackward(y, x, embedIdD.Weight);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class EmbedIDD
#else
    public static class EmbedIDF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> weight, IFunction<Real> embedId)
        {
            int outputCount = weight.Shape[1];

            Real[] result = new Real[x.Data.Length * outputCount];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < outputCount; j++)
                    {
                        result[i * outputCount + j + b * x.Length * outputCount] = weight.Data[(int)x.Data[b * x.Length + i] * outputCount + j];
                    }
                }
            }

            return NdArray.Convert(result, new[] { x.Length, outputCount }, x.BatchCount, embedId);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight)
        {
            int outputCount = weight.Shape[1];

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < outputCount; j++)
                    {
                        weight.Grad[(int)x.Data[b * x.Length + i] * outputCount + j] += y.Grad[i * outputCount + j + b * x.Length * outputCount];
                    }
                }
            }
        }
    }
}
