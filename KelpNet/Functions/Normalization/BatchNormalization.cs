using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Normalization
{
    //Chainerより移植　finetuningは未実装
    [Serializable]
    public class BatchNormalization : SingleInputFunction
    {
        public bool IsTrain;

        public NdArray Gamma;

        public NdArray Beta;

        public NdArray AvgMean;

        public NdArray AvgVar;


        private readonly Real Decay;
        private readonly Real Eps;

        private Real[] Std;
        private Real[] Xhat;

        private Real[] Mean;
        private Real[] Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, Array initialAvgMean = null, Array initialAvgVar = null, bool isTrain = true, string name = "BatchNorm") : base(name)
        {
            this.ChannelSize = channelSize;
            this.Decay = decay;
            this.Eps = eps;
            this.IsTrain = isTrain;

            this.Gamma = new NdArray(channelSize);
            this.Gamma.Data = Enumerable.Repeat((Real)1, channelSize).ToArray();
            this.Gamma.Name = this.Name + " Gamma";

            this.Beta = new NdArray(channelSize);
            this.Beta.Name = this.Name + " Beta";

            this.Parameters = new NdArray[this.IsTrain ? 2 : 4];

            //学習対象のParameterを登録
            this.Parameters[0] = this.Gamma;
            this.Parameters[1] = this.Beta;

            this.AvgMean = new NdArray(channelSize);
            this.AvgMean.Name = this.Name + " Mean";
            this.AvgVar = new NdArray(channelSize);
            this.AvgVar.Name = this.Name + " Variance";

            if (initialAvgMean != null)
            {
                this.AvgMean.Data = Real.GetArray(initialAvgMean);
            }

            if (initialAvgVar != null)
            {
                this.AvgVar.Data = Real.GetArray(initialAvgVar);
            }

            if (!this.IsTrain)
            {
                this.Parameters[2] = this.AvgMean;
                this.Parameters[3] = this.AvgVar;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray ForwardCpu(NdArray x)
        {
            //計算用パラメータの取得
            if (this.IsTrain)
            {
                //メンバのMeanとVarianceを設定する
                this.Variance = new Real[this.ChannelSize];
                for (int i = 0; i < this.Variance.Length; i++)
                {
                    this.Variance[i] = 0;
                }

                this.Mean = new Real[this.ChannelSize];
                for (int i = 0; i < this.Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        this.Mean[i] += x.Data[i + index * x.Length];
                    }

                    this.Mean[i] /= x.BatchCount;
                }

                for (int i = 0; i < this.Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        this.Variance[i] += (x.Data[i + index * x.Length] - this.Mean[i]) * (x.Data[i + index * x.Length] - this.Mean[i]);
                    }

                    this.Variance[i] /= x.BatchCount;
                }

                for (int i = 0; i < this.Variance.Length; i++)
                {
                    this.Variance[i] += this.Eps;
                }
            }
            else
            {
                this.Mean = this.AvgMean.Data;
                this.Variance = this.AvgVar.Data;
            }

            this.Std = new Real[this.Variance.Length];
            for (int i = 0; i < this.Variance.Length; i++)
            {
                this.Std[i] = Math.Sqrt(this.Variance[i]);
            }

            //結果を計算
            this.Xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];

            int dataSize = 1;
            for (int i = 1; i < x.Shape.Length; i++)
            {
                dataSize *= x.Shape[i];
            }

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = batchCount * this.ChannelSize * dataSize + i * dataSize + location;
                        this.Xhat[index] = (x.Data[index] - this.Mean[i]) / this.Std[i];
                        y[index] = this.Gamma.Data[i] * this.Xhat[index] + this.Beta.Data[i];
                    }
                }
            }

            //パラメータを更新
            if (this.IsTrain)
            {
                int m = x.BatchCount;
                Real adjust = m / Math.Max(m - 1.0, 1.0); // unbiased estimation

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

        private void BackwardCpu(NdArray y, NdArray x)
        {
            this.Beta.ClearGrad();
            this.Gamma.ClearGrad();

            for (int i = 0; i < this.ChannelSize; i++)
            {
                for (int j = 0; j < y.BatchCount; j++)
                {
                    this.Beta.Grad[i] += y.Grad[i + j * y.Length];
                    this.Gamma.Grad[i] += y.Grad[i + j * y.Length] * this.Xhat[j * this.ChannelSize + i];
                }
            }

            if (this.IsTrain)
            {
                // 学習あり
                int m = y.BatchCount;

                for (int i = 0; i < this.ChannelSize; i++)
                {
                    Real gs = this.Gamma.Data[i] / this.Std[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        Real val = (this.Xhat[j * this.ChannelSize + i] * this.Gamma.Grad[i] + this.Beta.Grad[i]) / m;

                        x.Grad[i + j * this.ChannelSize] += gs * (y.Grad[i + j * y.Length] - val);
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

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        x.Grad[i + j * this.ChannelSize] += gs * y.Grad[i + j * y.Length];
                    }
                }
            }
        }

        public override NdArray[] Predict(params NdArray[] input)
        {
            NdArray[] result;

            if (this.IsTrain)
            {
                //Predictはトレーニングしない
                this.IsTrain = false;

                result = this.Forward(input);

                //フラグをリセット
                this.IsTrain = true;
            }
            else
            {
                result = this.Forward(input);
            }

            return result;
        }
    }
}
