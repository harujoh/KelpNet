using System;
using System.Linq;
using System.Runtime.Serialization;
#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
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

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 2e-5, bool useGamma = true, bool useBeta = true, int initialGamma = 1, int initialBeta = 0, int? axis = null, int initialAvgMean = 0, int initialAvgVar = 1, bool train = true, bool finetune = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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

            switch (this)
            {
                case BatchNormalization<float> batchNormalizationF:
                    batchNormalizationF.Decay = (float)decay;
                    batchNormalizationF.Eps = (float)eps;

                    batchNormalizationF.Gamma.Data = Enumerable.Repeat((float)initialGamma, batchNormalizationF.Gamma.Data.Length).ToArray();
                    batchNormalizationF.Beta.Data = Enumerable.Repeat((float)initialBeta, batchNormalizationF.Beta.Data.Length).ToArray();

                    batchNormalizationF.AvgMean.Data = Enumerable.Repeat((float)initialAvgMean, batchNormalizationF.AvgMean.Data.Length).ToArray();
                    batchNormalizationF.AvgVar.Data = Enumerable.Repeat((float)initialAvgVar, batchNormalizationF.AvgVar.Data.Length).ToArray();
                    break;

                case BatchNormalization<double> batchNormalizationD:
                    batchNormalizationD.Decay = decay;
                    batchNormalizationD.Eps = eps;

                    batchNormalizationD.Gamma.Data = Enumerable.Repeat((double)initialGamma, batchNormalizationD.Gamma.Data.Length).ToArray();
                    batchNormalizationD.Beta.Data = Enumerable.Repeat((double)initialBeta, batchNormalizationD.Beta.Data.Length).ToArray();

                    batchNormalizationD.AvgMean.Data = Enumerable.Repeat((double)initialAvgMean, batchNormalizationD.AvgMean.Data.Length).ToArray();
                    batchNormalizationD.AvgVar.Data = Enumerable.Repeat((double)initialAvgVar, batchNormalizationD.AvgVar.Data.Length).ToArray();
                    break;
            }

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
                    batchNormalizationF.SingleInputForward = (x) => BatchNormalizationF.SingleInputForward(x, batchNormalizationF.Train, batchNormalizationF.Gamma, batchNormalizationF.Beta, batchNormalizationF.AvgMean, batchNormalizationF.AvgVar, ref batchNormalizationF.N, batchNormalizationF.Finetune, ref batchNormalizationF.Decay, batchNormalizationF.Eps, ref batchNormalizationF.Std, ref batchNormalizationF.Xhat, batchNormalizationF.ChannelSize, batchNormalizationF);
                    batchNormalizationF.SingleOutputBackward = (y, x) => BatchNormalizationF.SingleOutputBackward(y, x, batchNormalizationF.Train, batchNormalizationF.Gamma, batchNormalizationF.Beta, batchNormalizationF.AvgMean, batchNormalizationF.AvgVar, batchNormalizationF.Std, batchNormalizationF.Xhat, batchNormalizationF.ChannelSize);
                    break;
                case BatchNormalization<double> batchNormalizationD:
                    batchNormalizationD.SingleInputForward = (x) => BatchNormalizationD.SingleInputForward(x, batchNormalizationD.Train, batchNormalizationD.Gamma, batchNormalizationD.Beta, batchNormalizationD.AvgMean, batchNormalizationD.AvgVar, ref batchNormalizationD.N, batchNormalizationD.Finetune, ref batchNormalizationD.Decay, batchNormalizationD.Eps, ref batchNormalizationD.Std, ref batchNormalizationD.Xhat, batchNormalizationD.ChannelSize, batchNormalizationD);
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
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, bool Train, NdArray<Real> Gamma, NdArray<Real> Beta, NdArray<Real> AvgMean, NdArray<Real> AvgVar, ref Real N, bool Finetune, ref Real Decay, Real Eps, ref Real[] Std, ref Real[] Xhat, int ChannelSize, IFunction<Real> batchNorm)
        {
            Real[] Mean;
            Real[] Variance;

            if (Finetune)
            {
                N++;
                Decay = 1 - 1 / N;
            }

            int dataSize = x.Length / ChannelSize;

            //計算用パラメータの取得
            if (Train)
            {
                //メンバのMeanとVarianceを設定する
                Variance = new Real[ChannelSize];
                Mean = new Real[ChannelSize];

                for (int i = 0; i < ChannelSize; i++)
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
                Mean = AvgMean.Data;
                Variance = AvgVar.Data;
            }

            Std = new Real[ChannelSize];
            for (int i = 0; i < Std.Length; i++)
            {
                Std[i] = KelpMath.Sqrt(Variance[i] + Eps);
            }

            //結果を計算
            Xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < ChannelSize; i++)
            {
                for (int b = 0; b < x.BatchCount; b++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * x.Length + i * dataSize + location;
                        Xhat[index] = (x.Data[index] - Mean[i]) / Std[i];
                        y[index] = Gamma.Data[i] * Xhat[index] + Beta.Data[i];
                    }
                }
            }

            //パラメータを更新
            if (Train)
            {
                Real adjust = x.BatchCount / KelpMath.Max(x.BatchCount - 1, 1.0f); // unbiased estimation

                for (int i = 0; i < AvgMean.Data.Length; i++)
                {
                    AvgMean.Data[i] *= Decay;
                    Mean[i] *= 1 - Decay; // reuse buffer as a temporary
                    AvgMean.Data[i] += Mean[i];

                    AvgVar.Data[i] *= Decay;
                    Variance[i] *= (1 - Decay) * adjust; // reuse buffer as a temporary
                    AvgVar.Data[i] += Variance[i];
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, batchNorm);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, bool Train, NdArray<Real> Gamma, NdArray<Real> Beta, NdArray<Real> AvgMean, NdArray<Real> AvgVar, Real[] Std, Real[] Xhat, int ChannelSize)
        {
            Beta.InitGrad();
            Gamma.InitGrad();

            int dataSize = x.Length / ChannelSize;

            for (int i = 0; i < ChannelSize; i++)
            {
                for (int b = 0; b < x.BatchCount; b++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * y.Length + i * dataSize + location;
                        Beta.Grad[i] += y.Grad[index];
                        Gamma.Grad[i] += y.Grad[index] * Xhat[index];
                    }
                }
            }

            if (Train)
            {
                // 学習あり
                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];

                    for (int b = 0; b < y.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            int index = b * y.Length + i * dataSize + location;
                            Real val = (Xhat[index] * Gamma.Grad[i] + Beta.Grad[i]) / (y.BatchCount * dataSize);
                            x.Grad[index] += gs * (y.Grad[index] - val);
                        }
                    }
                }
            }
            else
            {
                // 学習なし
                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];
                    AvgMean.Grad[i] = -gs * Beta.Grad[i];
                    AvgVar.Grad[i] = -0.5f * Gamma.Data[i] / AvgVar.Data[i] * Gamma.Grad[i];

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
