using System;
using System.Linq;

namespace KelpNet
{
    //Chainerより移植　finetuningは未実装
    [Serializable]
    public class BatchNormalization<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "BatchNormalization";

        public bool IsTrain;

        public NdArray<T> Gamma;

        public NdArray<T> Beta;

        public NdArray<T> AvgMean;

        public NdArray<T> AvgVar;


        private readonly Real<T> Decay;
        private readonly Real<T> Eps;

        private RealArray<T> Std;
        private RealArray<T> Xhat;

        private RealArray<T> Mean;
        private RealArray<T> Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, Array initialAvgMean = null, Array initialAvgVar = null, bool isTrain = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.ChannelSize = channelSize;
            this.Decay = decay;
            this.Eps = eps;
            this.IsTrain = isTrain;

            this.Gamma = new NdArray<T>(channelSize);
            this.Gamma.Data = Enumerable.Repeat((Real<T>)1.0, channelSize).ToArray();
            this.Gamma.Name = this.Name + " Gamma";

            this.Beta = new NdArray<T>(channelSize);
            this.Beta.Name = this.Name + " Beta";

            this.Parameters = new NdArray<T>[this.IsTrain ? 2 : 4];

            //学習対象のParameterを登録
            this.Parameters[0] = this.Gamma;
            this.Parameters[1] = this.Beta;

            this.AvgMean = new NdArray<T>(channelSize);
            this.AvgMean.Name = this.Name + " Mean";
            this.AvgVar = new NdArray<T>(channelSize);
            this.AvgVar.Name = this.Name + " Variance";

            if (initialAvgMean != null)
            {
                this.AvgMean.Data = Real<T>.GetArray(initialAvgMean);
            }

            if (initialAvgVar != null)
            {
                this.AvgVar.Data = Real<T>.GetArray(initialAvgVar);
            }

            if (!this.IsTrain)
            {
                this.Parameters[2] = this.AvgMean;
                this.Parameters[3] = this.AvgVar;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray<T> ForwardCpu(NdArray<T> x)
        {
            //計算用パラメータの取得
            if (this.IsTrain)
            {
                //メンバのMeanとVarianceを設定する
                this.Variance = new T[this.ChannelSize];
                this.Mean = new T[this.ChannelSize];

                for (int i = 0; i < this.ChannelSize; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        for (int j = 0; j < x.Length; j++)
                        {
                            this.Mean[i] += x.Data[index * x.Length + i];
                        }
                    }

                    this.Mean[i] /= x.BatchCount;

                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        this.Variance[i] += (x.Data[index * x.Length + i] - this.Mean[i]) * (x.Data[index * x.Length + i] - this.Mean[i]);
                    }

                    this.Variance[i] /= x.BatchCount;
                    this.Variance[i] += this.Eps;
                }
            }
            else
            {
                this.Mean = this.AvgMean.Data;
                this.Variance = this.AvgVar.Data;
            }

            this.Std = new T[this.ChannelSize];

            for (int i = 0; i < this.ChannelSize; i++)
            {
                this.Std[i] = Math.Sqrt(this.Variance[i]);
            }

            //結果を計算
            this.Xhat = new T[x.DataLength];

            RealArray<T> y = new T[x.DataLength];

            int dataSize = 1;

            for (int i = 1; i < x.Shape.Length; i++)
            {
                dataSize *= x.Shape[i];
            }

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    int indexOffset = batchCount * this.ChannelSize * dataSize + i * dataSize;

                    for (int index = indexOffset; index < indexOffset + dataSize; index++)
                    {
                        this.Xhat[index] = (x.Data[index] - this.Mean[i]) / this.Std[i];
                        y[index] = this.Gamma.Data[i] * this.Xhat[index] + this.Beta.Data[i];
                    }
                }
            }

            //パラメータを更新
            if (this.IsTrain)
            {
                int m = x.BatchCount;
                Real<T> adjust = (m / Math.Max(m - 1.0, 1.0)); // unbiased estimation

                for (int i = 0; i < this.AvgMean.DataLength; i++)
                {
                    this.AvgMean.Data[i] *= this.Decay;
                    this.Mean[i] *= 1 - this.Decay; // reuse buffer as a temporary
                    this.AvgMean.Data[i] += this.Mean[i];

                    this.AvgVar.Data[i] *= this.Decay;
                    this.Variance[i] *= (1 - this.Decay) * adjust; // reuse buffer as a temporary
                    this.AvgVar.Data[i] += this.Variance[i];
                }
            }

            return NdArray<T>.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void BackwardCpu(NdArray<T> y, NdArray<T> x)
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
                    Real<T> gs = this.Gamma.Data[i] / this.Std[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        Real<T> val = (this.Xhat[j * this.ChannelSize + i] * this.Gamma.Grad[i] + this.Beta.Grad[i]) / m;

                        x.Grad[i + j * this.ChannelSize] += gs * (y.Grad[i + j * y.Length] - val);
                    }
                }
            }
            else
            {
                // 学習なし
                for (int i = 0; i < this.ChannelSize; i++)
                {
                    Real<T> gs = this.Gamma.Data[i] / this.Std[i];
                    this.AvgMean.Grad[i] = -gs * this.Beta.Grad[i];
                    this.AvgVar.Grad[i] = -0.5f * this.Gamma.Data[i] / this.AvgVar.Data[i] * this.Gamma.Grad[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        x.Grad[i + j * this.ChannelSize] += gs * y.Grad[i + j * y.Length];
                    }
                }
            }
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] input)
        {
            NdArray<T> result;

            if (this.IsTrain)
            {
                //Predictはトレーニングしない
                this.IsTrain = false;

                result = this.SingleInputForward(input[0]);

                //フラグをリセット
                this.IsTrain = true;
            }
            else
            {
                result = this.SingleInputForward(input[0]);
            }

            return new[] { result };
        }
    }
}
