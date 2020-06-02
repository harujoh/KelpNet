using System;
using System.Linq;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#elif NETSTANDARD2_0
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class BatchNormalization<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "BatchNormalization";

        public bool Train;

        public NdArray<T> Gamma;

        public NdArray<T> Beta;

        public NdArray<T> AvgMean;

        public NdArray<T> AvgVar;

        private T N;
        private bool Finetune;

        private T Decay;
        private T Eps;

        [NonSerialized]
        private T[] Std;

        [NonSerialized]
        private T[] Xhat;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, T? decay = null, T? eps = null, bool useGamma = true, bool useBeta = true, int initialGamma = 1, int initialBeta = 0, int initialAvgMean = 0, int initialAvgVar = 1, bool train = true, bool finetune = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.ChannelSize = channelSize;
            this.Train = train;

            this.Finetune = finetune;

            this.Gamma = new NdArray<T>(channelSize);
            this.Gamma.Name = this.Name + " Gamma";

            this.Beta = new NdArray<T>(channelSize);
            this.Beta.Name = this.Name + " Beta";

            int paramIndex = 0;
            int paramCount = 0;

            if (useGamma) paramCount++;
            if (useBeta) paramCount++;
            if (!train) paramCount += 2;

            this.Parameters = new NdArray<T>[paramCount];

            //学習対象のParameterを登録
            if (useGamma) this.Parameters[paramIndex++] = this.Gamma;
            if (useBeta) this.Parameters[paramIndex++] = this.Beta;

            this.AvgMean = new NdArray<T>(channelSize);
            this.AvgMean.Name = this.Name + " Mean";

            this.AvgVar = new NdArray<T>(channelSize);
            this.AvgVar.Name = this.Name + " Variance";

            this.Decay = decay ?? (TVal<T>)0.9;
            this.Eps = eps ?? (TVal<T>)2e-5;

            this.Gamma.Fill((TVal<T>)initialGamma);
            this.Beta.Fill((TVal<T>)initialBeta);

            this.AvgMean.Fill((TVal<T>)initialAvgMean);
            this.AvgVar.Fill((TVal<T>)initialAvgVar);


            //自分で学習せずオプティマイザに任せる
            if (!this.Train)
            {
                this.Parameters[paramIndex++] = this.AvgMean;
                this.Parameters[paramIndex] = this.AvgVar;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.Predict = this.BatchNormalizationPredict;

            switch (this)
            {
                case BatchNormalization<float> batchNormalizationF:
                    batchNormalizationF.SingleInputForward = (x) => BatchNormalizationF.SingleInputForward(x, batchNormalizationF.Train, batchNormalizationF.Gamma, batchNormalizationF.Beta, batchNormalizationF.AvgMean, batchNormalizationF.AvgVar, ref batchNormalizationF.N, batchNormalizationF.Finetune, ref batchNormalizationF.Decay, batchNormalizationF.Eps, out batchNormalizationF.Std, out batchNormalizationF.Xhat, batchNormalizationF.ChannelSize, batchNormalizationF);
                    batchNormalizationF.SingleOutputBackward = (y, x) => BatchNormalizationF.SingleOutputBackward(y, x, batchNormalizationF.Train, batchNormalizationF.Gamma, batchNormalizationF.Beta, batchNormalizationF.AvgMean, batchNormalizationF.AvgVar, batchNormalizationF.Std, batchNormalizationF.Xhat, batchNormalizationF.ChannelSize);
                    break;
                case BatchNormalization<double> batchNormalizationD:
                    batchNormalizationD.SingleInputForward = (x) => BatchNormalizationD.SingleInputForward(x, batchNormalizationD.Train, batchNormalizationD.Gamma, batchNormalizationD.Beta, batchNormalizationD.AvgMean, batchNormalizationD.AvgVar, ref batchNormalizationD.N, batchNormalizationD.Finetune, ref batchNormalizationD.Decay, batchNormalizationD.Eps, out batchNormalizationD.Std, out batchNormalizationD.Xhat, batchNormalizationD.ChannelSize, batchNormalizationD);
                    batchNormalizationD.SingleOutputBackward = (y, x) => BatchNormalizationD.SingleOutputBackward(y, x, batchNormalizationD.Train, batchNormalizationD.Gamma, batchNormalizationD.Beta, batchNormalizationD.AvgMean, batchNormalizationD.AvgVar, batchNormalizationD.Std, batchNormalizationD.Xhat, batchNormalizationD.ChannelSize);
                    break;
            }
        }

        public NdArray<T>[] BatchNormalizationPredict(params NdArray<T>[] input)
        {
            NdArray<T> result;

            if (Train)
            {
                //Predictはトレーニングしない
                Train = false;

                result = this.SingleInputForward(input[0]);

                //フラグをリセット
                Train = true;
            }
            else
            {
                result = this.SingleInputForward(input[0]);
            }

            return new[] { result };
        }
    }
#endif

#if DOUBLE
    public static class BatchNormalizationD
#else
    public static class BatchNormalizationF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, bool train, NdArray<Real> gamma, NdArray<Real> beta, NdArray<Real> avgMean, NdArray<Real> avgVar, ref Real n, bool finetune, ref Real decay, Real Eps, out Real[] std, out Real[] xhat, int channelSize, IFunction<Real> batchNorm)
        {
            Real[] Mean;
            Real[] Variance;

            if (finetune)
            {
                n++;
                decay = 1 - 1 / n;
            }

            int dataSize = x.Length / channelSize;

            //計算用パラメータの取得
            if (train)
            {
                //メンバのMeanとVarianceを設定する
                Variance = new Real[channelSize];
                Mean = new Real[channelSize];

                for (int i = 0; i < channelSize; i++)
                {
                    for (int b = 0; b < x.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            Mean[i] += x.Data[b * x.Length + i * dataSize + location];
                        }
                    }

                    Mean[i] /= x.BatchCount * dataSize;

                    for (int b = 0; b < x.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            Variance[i] += (x.Data[b * x.Length + i * dataSize + location] - Mean[i]) * (x.Data[b * x.Length + i * dataSize + location] - Mean[i]);
                        }
                    }

                    Variance[i] /= x.BatchCount * dataSize;
                }
            }
            else
            {
                Mean = avgMean.Data;
                Variance = avgVar.Data;
            }

            std = new Real[channelSize];
            for (int i = 0; i < std.Length; i++)
            {
                std[i] = Math.Sqrt(Variance[i] + Eps);
            }

            //結果を計算
            xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];


            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < channelSize; i++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * x.Length + i * dataSize + location;
                        xhat[index] = (x.Data[index] - Mean[i]) / std[i];
                        y[index] = gamma.Data[i] * xhat[index] + beta.Data[i];
                    }
                }
            }

            //パラメータを更新
            if (train)
            {
                Real adjust = x.BatchCount / Math.Max(x.BatchCount - 1, 1.0f); // unbiased estimation

                for (int i = 0; i < avgMean.Data.Length; i++)
                {
                    avgMean.Data[i] *= decay;
                    Mean[i] *= 1 - decay; // reuse buffer as a temporary
                    avgMean.Data[i] += Mean[i];

                    avgVar.Data[i] *= decay;
                    Variance[i] *= (1 - decay) * adjust; // reuse buffer as a temporary
                    avgVar.Data[i] += Variance[i];
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, batchNorm);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, bool train, NdArray<Real> gamma, NdArray<Real> beta, NdArray<Real> avgMean, NdArray<Real> avgVar, Real[] std, Real[] xhat, int channelSize)
        {
            beta.InitGrad();
            gamma.InitGrad();

            int dataSize = x.Length / channelSize;

            for (int i = 0; i < channelSize; i++)
            {
                for (int b = 0; b < x.BatchCount; b++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * y.Length + i * dataSize + location;
                        beta.Grad[i] += y.Grad[index];
                        gamma.Grad[i] += y.Grad[index] * xhat[index];
                    }
                }
            }

            if (train)
            {
                // 学習あり
                for (int i = 0; i < channelSize; i++)
                {
                    Real gs = gamma.Data[i] / std[i];

                    for (int b = 0; b < y.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            int index = b * y.Length + i * dataSize + location;
                            Real val = (xhat[index] * gamma.Grad[i] + beta.Grad[i]) / (y.BatchCount * dataSize);
                            x.Grad[index] += gs * (y.Grad[index] - val);
                        }
                    }
                }
            }
            else
            {
                // 学習なし
                for (int i = 0; i < channelSize; i++)
                {
                    Real gs = gamma.Data[i] / std[i];
                    avgMean.Grad[i] = -gs * beta.Grad[i];
                    avgVar.Grad[i] = -0.5f * gamma.Data[i] / avgVar.Data[i] * gamma.Grad[i];

                    for (int b = 0; b < y.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            x.Grad[b * y.Length + i * dataSize + location] += gs * y.Grad[b * y.Length + i * dataSize + location];
                        }
                    }
                }
            }
        }
    }
}
