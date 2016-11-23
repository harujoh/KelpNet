using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Normalization
{
    //Chainerより移植　finetuningは未実装
    [Serializable]
    public class BatchNormalization : Function
    {
        private bool IsTrain;
        private readonly bool InputIsTrain;
        private readonly NdArray Gamma;
        private readonly NdArray gGamma;
        private readonly NdArray Beta;
        private readonly NdArray gBeta;
        private readonly double Decay;
        private readonly double Eps;
        private readonly NdArray AvgMean;
        private readonly NdArray AvgVar;
        private readonly NdArray gMean;
        private readonly NdArray gVariance;
        private double[] Std;
        private double[,] Xhat;

        private double[] Mean;
        private double[] Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, bool isTrain = true, string name = "BatchNorm") : base(name)
        {
            this.Gamma = NdArray.Ones(channelSize);
            this.Beta = NdArray.Zeros(channelSize);

            this.gGamma = NdArray.ZerosLike(this.Gamma);
            this.gBeta = NdArray.ZerosLike(this.Beta);

            //学習対象のParameterを登録
            Parameters.Add(new OptimizeParameter(this.Gamma, this.gGamma, Name + " Gamma"));
            Parameters.Add(new OptimizeParameter(this.Beta, this.gBeta, Name + " Beta"));

            this.IsTrain = isTrain;
            this.InputIsTrain = isTrain;

            this.AvgMean = NdArray.Zeros(channelSize);
            this.AvgVar = NdArray.Zeros(channelSize);

            if (!this.IsTrain)
            {
                this.gMean = NdArray.Zeros(channelSize);
                this.gVariance = NdArray.Zeros(channelSize);
                Parameters.Add(new OptimizeParameter(this.AvgMean, this.gMean, Name + " Mean"));
                Parameters.Add(new OptimizeParameter(this.AvgVar, this.gVariance, Name + " Variance"));
            }

            this.Decay = decay;
            this.Eps = eps;

            this.ChannelSize = channelSize;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] y = new NdArray[x.Length];

            //計算用パラメータの取得
            if (this.IsTrain)
            {
                //メンバのMeanとVarianceを設定する
                this.CalcVariance(x);
            }
            else
            {
                this.Mean = this.AvgMean.Data;
                this.Variance = this.AvgVar.Data;
            }

            this.Std = new double[this.Variance.Length];
            for (int i = 0; i < this.Variance.Length; i++)
            {
                this.Std[i] = Math.Sqrt(this.Variance[i]);
            }

            //結果を計算
            this.Xhat = new double[x.Length, this.ChannelSize];
#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                y[i] = NdArray.ZerosLike(x[i]);

                for (int j = 0; j < this.ChannelSize; j++)
                {
                    this.Xhat[i,j] = (x[i].Data[j] - this.Mean[j]) / this.Std[j];
                    y[i].Data[j] = this.Gamma.Data[j] * this.Xhat[i,j] + this.Beta.Data[j];
                }
            }
#if !DEBUG
            );
#endif

            //パラメータを更新
            if (this.IsTrain)
            {
                var m = x.Length;
                var adjust = m / Math.Max(m - 1.0, 1.0); // unbiased estimation

                for (int i = 0; i < this.AvgMean.Length; i++)
                {
                    this.AvgMean.Data[i] *= this.Decay;
                    this.Mean[i] *= 1 - this.Decay; // reuse buffer as a temporary
                    this.AvgMean.Data[i] += this.Mean[i];

                    this.AvgVar.Data[i] *= this.Decay;
                    this.Variance[i] *= (1 - this.Decay) * adjust; // reuse buffer as a temporary
                    this.AvgVar.Data[i] += this.Variance[i];
                }
            }

            return y;
        }

        public void CalcVariance(params NdArray[] values)
        {
            this.Variance = new double[this.ChannelSize];
            for (int i = 0; i < this.Variance.Length; i++)
            {
                this.Variance[i] = 0;
            }

            this.Mean = new double[this.ChannelSize];
            for (int i = 0; i < this.Mean.Length; i++)
            {
                foreach (NdArray value in values)
                {
                    this.Mean[i] += value.Data[i];
                }

                this.Mean[i] /= values.Length;
            }


            for (int i = 0; i < this.Mean.Length; i++)
            {
                foreach (NdArray value in values)
                {
                    this.Variance[i] += Math.Pow(value.Data[i] - this.Mean[i], 2);
                }

                this.Variance[i] /= values.Length;
            }

            for (int i = 0; i < this.Variance.Length; i++)
            {
                this.Variance[i] += this.Eps;
            }
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            NdArray[] gx = new NdArray[gy.Length];
            for (int i = 0; i < gy.Length; i++)
            {
                gx[i] = NdArray.Zeros(this.ChannelSize);
            }

            this.gBeta.Fill(0);
            this.gGamma.Fill(0);

#if DEBUG
            for (int i = 0; i < this.ChannelSize; i++)
#else
            Parallel.For(0, this.ChannelSize, i =>
#endif
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gBeta.Data[i] += gy[j].Data[i];
                    this.gGamma.Data[i] += gy[j].Data[i] * this.Xhat[j,i];
                }
            }
#if !DEBUG
            );
#endif

            if (!this.IsTrain)
            {
                // 学習なし
#if DEBUG
                for (int i = 0; i < this.ChannelSize; i++)
#else
                Parallel.For(0, this.ChannelSize, i =>
#endif
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];
                    this.gMean.Data[i] = -gs * this.gBeta.Data[i];
                    this.gVariance.Data[i] = -0.5 * this.Gamma.Data[i] / this.AvgVar.Data[i] * this.gGamma.Data[i];

                    for (int j = 0; j < gy.Length; j++)
                    {
                        gx[j].Data[i] = gs * gy[j].Data[i];
                    }
                }
#if !DEBUG
                );
#endif
            }
            else
            {
                var m = gy.Length;

#if DEBUG
                for (int i = 0; i < this.ChannelSize; i++)
#else
                Parallel.For(0, this.ChannelSize, i =>
#endif
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];

                    for (int j = 0; j < gy.Length; j++)
                    {
                        var val = (this.Xhat[j,i] * this.gGamma.Data[i] + this.gBeta.Data[i]) / m;

                        gx[j].Data[i] = gs * (gy[j].Data[i] - val);
                    }
                }
#if !DEBUG
                );
#endif
            }

            return gx;
        }

        public override NdArray[] Predict(NdArray[] input)
        {
            this.IsTrain = false;

            var result = this.ForwardSingle(input);

            //フラグをリセット
            this.IsTrain = this.InputIsTrain;

            return result;
        }
    }
}
