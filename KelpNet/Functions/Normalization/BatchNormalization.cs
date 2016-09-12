using System;

namespace KelpNet.Functions.Normalization
{
    //Chainerより移植　finetuningは未実装
    public class BatchNormalization : PredictableFunction, IBatchFunction
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
        private double[][] Xhat;

        private double[] Mean;
        private double[] Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, bool isTrain = true)
        {
            this.Gamma = NdArray.Ones(channelSize);
            this.Beta = NdArray.Zeros(channelSize);

            this.gGamma = NdArray.ZerosLike(this.Gamma);
            this.gBeta = NdArray.ZerosLike(this.Beta);

            //学習対象のParameterを登録
            Parameters.Add(new Parameter(this.Gamma, this.gGamma));
            Parameters.Add(new Parameter(this.Beta, this.gBeta));

            this.IsTrain = isTrain;
            this.InputIsTrain = isTrain;

            this.AvgMean = NdArray.Zeros(channelSize);
            this.AvgVar = NdArray.Zeros(channelSize);

            if (!this.IsTrain)
            {
                this.gMean = NdArray.Zeros(channelSize);
                this.gVariance = NdArray.Zeros(channelSize);
                Parameters.Add(new Parameter(this.AvgMean, this.gMean));
                Parameters.Add(new Parameter(this.AvgVar, this.gVariance));
            }

            this.Decay = decay;
            this.Eps = eps;

            this.ChannelSize = channelSize;
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

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
            this.Xhat = new[]
            {
                new double[this.ChannelSize]
            };

            for (int i = 0; i < x.Length; i++)
            {
                this.Xhat[0][i] = (x.Data[i] - this.Mean[i]) / this.Std[i];
                y.Data[i] = this.Gamma.Data[i] * this.Xhat[0][i] + this.Beta.Data[i];
            }

            return y;
        }

        public NdArray[] BatchForward(NdArray[] x)
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
            this.Xhat = new double[x.Length][];
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = NdArray.EmptyLike(x[i]);
                this.Xhat[i] = new double[this.ChannelSize];

                for (int j = 0; j < this.ChannelSize; j++)
                {
                    this.Xhat[i][j] = (x[i].Data[j] - this.Mean[j]) / this.Std[j];
                    y[i].Data[j] = this.Gamma.Data[j] * this.Xhat[i][j] + this.Beta.Data[j];
                }
            }

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
            for (int j = 0; j < this.Mean.Length; j++)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    this.Mean[j] += values[i].Data[j];
                }

                this.Mean[j] /= values.Length;
            }


            for (int j = 0; j < this.Mean.Length; j++)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    this.Variance[j] += Math.Pow(values[i].Data[j] - this.Mean[j], 2);
                }

                this.Variance[j] /= values.Length;
            }

            for (int i = 0; i < this.Variance.Length; i++)
            {
                this.Variance[i] += this.Eps;
            }
        }

        protected override NdArray BackwardSingle(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray gx = NdArray.EmptyLike(gy);

            this.gBeta.Fill(0);
            this.gGamma.Fill(0);

            for (int j = 0; j < this.ChannelSize; j++)
            {
                this.gBeta.Data[j] += gy.Data[j];
                this.gGamma.Data[j] += gy.Data[j] * this.Xhat[0][j];
            }

            if (!this.IsTrain)
            {
                // 学習なし
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];
                    this.gMean.Data[i] = -gs * this.gBeta.Data[i];
                    this.gVariance.Data[i] = -0.5 * this.Gamma.Data[i] / this.AvgVar.Data[i] * this.gGamma.Data[i];
                    gx.Data[i] = gs * gy.Data[i];
                }
            }
            else
            {
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];
                    var val = this.Xhat[0][i] * this.gGamma.Data[i] + this.gBeta.Data[i];
                    gx.Data[i] = gs * (gy.Data[i] - val);
                }
            }

            return gx;
        }

        public NdArray[] BatchBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput)
        {
            NdArray[] gx = new NdArray[gy.Length];
            for (int i = 0; i < gy.Length; i++)
            {
                gx[i] = NdArray.Empty(this.ChannelSize);
            }

            this.gBeta.Fill(0);
            this.gGamma.Fill(0);

            for (int i = 0; i < gy.Length; i++)
            {
                for (int j = 0; j < this.ChannelSize; j++)
                {
                    this.gBeta.Data[j] += gy[i].Data[j];
                    this.gGamma.Data[j] += gy[i].Data[j] * this.Xhat[i][j];
                }
            }

            if (!this.IsTrain)
            {
                // 学習なし
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];
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
                var m = gy.Length;

                for (int i = 0; i < this.ChannelSize; i++)
                {
                    var gs = this.Gamma.Data[i] / this.Std[i];

                    for (int j = 0; j < gy.Length; j++)
                    {
                        var val = (this.Xhat[j][i] * this.gGamma.Data[i] + this.gBeta.Data[i]) / m;

                        gx[j].Data[i] = gs * (gy[j].Data[i] - val);
                    }
                }
            }

            return gx;
        }

        public override NdArray Predict(NdArray input)
        {
            this.IsTrain = false;

            var result = this.ForwardSingle(input);

            //フラグをリセット
            this.IsTrain = this.InputIsTrain;

            return result;
        }
    }
}
