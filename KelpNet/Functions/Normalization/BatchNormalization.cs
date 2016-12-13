using System;
using KelpNet.Common;
using System.Threading.Tasks;

namespace KelpNet.Functions.Normalization
{
    //Chainerより移植　finetuningは未実装
    [Serializable]
    public class BatchNormalization : Function
    {
        private bool IsTrain;
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

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, bool isTrain = true, string name = "BatchNorm", bool isParallel = true) : base(name, isParallel)
        {
            this.ChannelSize = channelSize;
            this.Decay = decay;
            this.Eps = eps;
            this.IsTrain = isTrain;

            this.Gamma = NdArray.Ones(channelSize);
            this.Beta = NdArray.Zeros(channelSize);

            this.gGamma = NdArray.ZerosLike(this.Gamma);
            this.gBeta = NdArray.ZerosLike(this.Beta);

            this.Parameters = new FunctionParameter[this.IsTrain ? 2 : 4];

            //学習対象のParameterを登録
            this.Parameters[0] = new FunctionParameter(this.Gamma, this.gGamma, this.Name + " Gamma");
            this.Parameters[1] = new FunctionParameter(this.Beta, this.gBeta, this.Name + " Beta");

            this.AvgMean = NdArray.Zeros(channelSize);
            this.AvgVar = NdArray.Zeros(channelSize);

            if (!this.IsTrain)
            {
                this.gMean = NdArray.Zeros(channelSize);
                this.gVariance = NdArray.Zeros(channelSize);

                this.Parameters[2] = new FunctionParameter(this.AvgMean, this.gMean, this.Name + " Mean");
                this.Parameters[3] = new FunctionParameter(this.AvgVar, this.gVariance, this.Name + " Variance");
            }
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

            if (IsParallel)
            {
                Parallel.For(0, x.Length, i =>
                {
                    y[i] = NdArray.ZerosLike(x[i]);

                    for (int j = 0; j < this.ChannelSize; j++)
                    {
                        this.Xhat[i, j] = (x[i].Data[j] - this.Mean[j]) / this.Std[j];
                        y[i].Data[j] = this.Gamma.Data[j] * this.Xhat[i, j] + this.Beta.Data[j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i] = NdArray.ZerosLike(x[i]);

                    for (int j = 0; j < this.ChannelSize; j++)
                    {
                        this.Xhat[i, j] = (x[i].Data[j] - this.Mean[j]) / this.Std[j];
                        y[i].Data[j] = this.Gamma.Data[j] * this.Xhat[i, j] + this.Beta.Data[j];
                    }
                }
            }

            //パラメータを更新
            if (this.IsTrain)
            {
                int m = x.Length;
                double adjust = m / Math.Max(m - 1.0, 1.0); // unbiased estimation

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

            for (int i = 0; i < this.ChannelSize; i++)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gBeta.Data[i] += gy[j].Data[i];
                    this.gGamma.Data[i] += gy[j].Data[i] * this.Xhat[j, i];
                }
            }

            if (!this.IsTrain)
            {
                // 学習なし
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    double gs = this.Gamma.Data[i] / this.Std[i];
                    this.gMean.Data[i] = -gs * this.gBeta.Data[i];
                    this.gVariance.Data[i] = -0.5 * this.Gamma.Data[i] / this.AvgVar.Data[i] * this.gGamma.Data[i];

                    for (int j = 0; j < gy.Length; j++)
                    {
                        gx[j].Data[i] = gs * gy[j].Data[i];
                    }
                }
            }
            else
            {
                int m = gy.Length;

                for (int i = 0; i < this.ChannelSize; i++)
                {
                    double gs = this.Gamma.Data[i] / this.Std[i];

                    for (int j = 0; j < gy.Length; j++)
                    {
                        double val = (this.Xhat[j, i] * this.gGamma.Data[i] + this.gBeta.Data[i]) / m;

                        gx[j].Data[i] = gs * (gy[j].Data[i] - val);
                    }
                }
            }

            return gx;
        }

        public override NdArray[] Predict(NdArray[] input)
        {
            NdArray[] result;

            if (this.IsTrain)
            {
                //Predictはトレーニングしない
                this.IsTrain = false;

                result = this.ForwardSingle(input);

                //フラグをリセット
                this.IsTrain = true;
            }
            else
            {
                result = this.ForwardSingle(input);
            }

            return result;
        }
    }
}
