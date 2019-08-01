using System;
using System.Linq;

namespace KelpNet
{
    [Serializable]
    public class BatchNormalization : SingleInputFunction
    {
        const string FUNCTION_NAME = "BatchNormalization";

        public bool Train;

        public NdArray Gamma;

        public NdArray Beta;

        public NdArray AvgMean;

        public NdArray AvgVar;

        private int N = 0;
        private bool Finetune;

        private Real Decay;
        private Real Eps;

        [NonSerialized]
        private Real[] Std;

        [NonSerialized]
        private Real[] Xhat;

        private Real[] Mean;
        private Real[] Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 2e-5, bool useGamma = true, bool useBeta = true, int initialGamma = 1, int initialBeta = 0, int? axis = null, int initialAvgMean = 0, int initialAvgVar = 1, bool train = true, bool finetune = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.ChannelSize = channelSize;
            this.Decay = decay;
            this.Eps = eps;
            this.Train = train;

            this.Finetune = finetune;

            this.Gamma = new NdArray(channelSize);
            this.Gamma.Name = this.Name + " Gamma";

            this.Beta = new NdArray(channelSize);
            this.Beta.Name = this.Name + " Beta";

            int paramIndex = 0;
            int paramCount = 0;

            if (useGamma) paramCount++;
            if (useBeta) paramCount++;
            if (!train) paramCount += 2;

            this.Parameters = new NdArray[paramCount];

            //学習対象のParameterを登録
            if (useGamma) this.Parameters[paramIndex++] = this.Gamma;
            if (useBeta) this.Parameters[paramIndex++] = this.Beta;

            this.AvgMean = new NdArray(channelSize);
            this.AvgMean.Name = this.Name + " Mean";

            this.AvgVar = new NdArray(channelSize);
            this.AvgVar.Name = this.Name + " Variance";

            this.Gamma.Data = Enumerable.Repeat((Real)initialGamma, this.Gamma.Data.Length).ToArray();
            this.Beta.Data = Enumerable.Repeat((Real)initialBeta, this.Beta.Data.Length).ToArray();

            this.AvgMean.Data = Enumerable.Repeat((Real)initialAvgMean, this.AvgMean.Data.Length).ToArray();
            this.AvgVar.Data = Enumerable.Repeat((Real)initialAvgVar, this.AvgVar.Data.Length).ToArray();

            if (!this.Train)
            {
                this.Parameters[paramIndex++] = this.AvgMean;
                this.Parameters[paramIndex] = this.AvgVar;
            }
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            if (Finetune)
            {
                N++;
                Decay = 1 - 1 / N;
            }

            int dataSize = x.Length / ChannelSize;

            //計算用パラメータの取得
            if (this.Train)
            {
                //メンバのMeanとVarianceを設定する
                this.Variance = new Real[this.ChannelSize];
                this.Mean = new Real[this.ChannelSize];

                for (int i = 0; i < this.ChannelSize; i++)
                {
                    for (int b = 0; b < x.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            this.Mean[i] += x.Data[b * x.Length + i * dataSize + location];
                        }
                    }

                    this.Mean[i] /= x.BatchCount * dataSize;

                    for (int b = 0; b < x.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            this.Variance[i] += (x.Data[b * x.Length + i * dataSize + location] - this.Mean[i]) * (x.Data[b * x.Length + i * dataSize + location] - this.Mean[i]);
                        }
                    }

                    this.Variance[i] /= x.BatchCount * dataSize;
                }
            }
            else
            {
                this.Mean = this.AvgMean.Data;
                this.Variance = this.AvgVar.Data;
            }

            this.Std = new Real[this.ChannelSize];
            for (int i = 0; i < this.Std.Length; i++)
            {
                this.Std[i] = Math.Sqrt(this.Variance[i] + this.Eps);
            }

            //結果を計算
            this.Xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < this.ChannelSize; i++)
            {
                for (int b = 0; b < x.BatchCount; b++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * x.Length + i * dataSize + location;
                        this.Xhat[index] = (x.Data[index] - this.Mean[i]) / this.Std[i];
                        y[index] = this.Gamma.Data[i] * this.Xhat[index] + this.Beta.Data[i];
                    }
                }
            }

            //パラメータを更新
            if (this.Train)
            {
                Real adjust = x.BatchCount / Math.Max(x.BatchCount - 1.0, 1.0); // unbiased estimation

                for (int i = 0; i < this.AvgMean.Data.Length; i++)
                {
                    this.AvgMean.Data[i] *= this.Decay;
                    this.Mean[i] *= 1 - this.Decay; // reuse buffer as a temporary
                    this.AvgMean.Data[i] += this.Mean[i];

                    this.AvgVar.Data[i] *= this.Decay;
                    this.Variance[i] *= (1 - this.Decay) * adjust; // reuse buffer as a temporary
                    this.AvgVar.Data[i] += this.Variance[i];
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            this.Beta.InitGrad();
            this.Gamma.InitGrad();

            int dataSize = x.Length / ChannelSize;

            for (int i = 0; i < this.ChannelSize; i++)
            {
                for (int b = 0; b < x.BatchCount; b++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = b * y.Length + i * dataSize + location;
                        this.Beta.Grad[i] += y.Grad[index];
                        this.Gamma.Grad[i] += y.Grad[index] * this.Xhat[index];
                    }
                }
            }

            if (this.Train)
            {
                // 学習あり
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    Real gs = this.Gamma.Data[i] / this.Std[i];

                    for (int b = 0; b < y.BatchCount; b++)
                    {
                        for (int location = 0; location < dataSize; location++)
                        {
                            int index = b * y.Length + i * dataSize + location;
                            Real val = (this.Xhat[index] * this.Gamma.Grad[i] + this.Beta.Grad[i]) / (y.BatchCount * dataSize);
                            x.Grad[index] += gs * (y.Grad[index] - val);
                        }
                    }
                }
            }
            else
            {
                // 学習なし
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    Real gs = this.Gamma.Data[i] / this.Std[i];
                    this.AvgMean.Grad[i] = -gs * this.Beta.Grad[i];
                    this.AvgVar.Grad[i] = -0.5 * this.Gamma.Data[i] / this.AvgVar.Data[i] * this.Gamma.Grad[i];

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

        public override NdArray[] Predict(params NdArray[] input)
        {
            NdArray result;

            if (this.Train)
            {
                //Predictはトレーニングしない
                this.Train = false;

                result = this.SingleInputForward(input[0]);

                //フラグをリセット
                this.Train = true;
            }
            else
            {
                result = this.SingleInputForward(input[0]);
            }

            return new[] { result };
        }
    }
}
