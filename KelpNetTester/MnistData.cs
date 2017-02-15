using System;
using KelpNet.Common;
using MNISTLoader;

namespace KelpNetTester
{
    class MnistData
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();

        private NdArray[] X;
        private NdArray[] Tx;

        private NdArray[] Y;
        private NdArray[] Ty;

        public MnistData()
        {
            //トレーニングデータ
            this.X = new NdArray[this.mnistDataLoader.TrainData.Length];
            //トレーニングデータラベル
            this.Tx = new NdArray[this.mnistDataLoader.TrainData.Length];

            for (int i = 0; i < this.mnistDataLoader.TrainData.Length; i++)
            {
                double[] x = new double[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = this.mnistDataLoader.TrainData[i][j] / 255.0;
                }
                this.X[i] = new NdArray(x, new[] { 1, 28, 28 });

                this.Tx[i] = NdArray.FromArray(new[] { (int)this.mnistDataLoader.TrainLabel[i] });
            }

            //教師データ
            this.Y = new NdArray[this.mnistDataLoader.TeachData.Length];
            //教師データラベル
            this.Ty = new NdArray[this.mnistDataLoader.TeachData.Length];

            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                double[] y = new double[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[j] = this.mnistDataLoader.TeachData[i][j] / 255.0;
                }
                this.Y[i] = new NdArray(y, new[] { 1, 28, 28 });

                this.Ty[i] = NdArray.FromArray(new[] { (int)this.mnistDataLoader.TeachLabel[i] });
            }
        }

        public MnistDataSet GetRandomYSet(int dataCount)
        {
            BatchArray listY = new BatchArray(new[] { 1, 28, 28 }, dataCount);
            BatchArray listTy = new BatchArray(new[] { 1 }, dataCount);

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                Buffer.BlockCopy(this.Y[index].Data, 0, listY.Data, sizeof(double) * j * listY.Length, sizeof(double) * listY.Length);
                listTy.Data[j] = this.Ty[index].Data[0];
            }

            return new MnistDataSet(listY, listTy);
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            BatchArray listX = new BatchArray(new[] { 1, 28, 28 }, dataCount);
            BatchArray listTx = new BatchArray(new[] { 1 }, dataCount);

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.X.Length);

                Buffer.BlockCopy(this.X[index].Data, 0, listX.Data, sizeof(double) * j * listX.Length, sizeof(double) * listX.Length);
                listTx.Data[j] = this.Tx[index].Data[0];
            }

            return new MnistDataSet(listX, listTx);
        }
    }

    public class MnistDataSet
    {
        public BatchArray Data;
        public BatchArray Label;

        public MnistDataSet(BatchArray data, BatchArray label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
