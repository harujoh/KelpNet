using System;
using KelpNet.Tools;

namespace KelpNet.Sample
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
                Real[] x = new Real[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = this.mnistDataLoader.TrainData[i][j] / 255.0;
                }
                this.X[i] = new NdArray(x, new[] { 1, 28, 28 });

                this.Tx[i] = new NdArray(new[] { (Real)this.mnistDataLoader.TrainLabel[i] });
            }

            //教師データ
            this.Y = new NdArray[this.mnistDataLoader.TeachData.Length];
            //教師データラベル
            this.Ty = new NdArray[this.mnistDataLoader.TeachData.Length];

            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                Real[] y = new Real[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[j] = this.mnistDataLoader.TeachData[i][j] / 255.0;
                }
                this.Y[i] = new NdArray(y, new[] { 1, 28, 28 });

                this.Ty[i] = new NdArray(new[] { (Real)this.mnistDataLoader.TeachLabel[i] });
            }
        }

        public TestDataSet GetRandomYSet(int dataCount)
        {
            NdArray listY = new NdArray(new[] { 1, 28, 28 }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                Array.Copy(this.Y[index].Data, 0, listY.Data,i * listY.Length,listY.Length);
                listTy.Data[i] = this.Ty[index].Data[0];
            }

            return new TestDataSet(listY, listTy);
        }

        public TestDataSet GetRandomXSet(int dataCount)
        {
            NdArray listX = new NdArray(new[] { 1, 28, 28 }, dataCount);
            NdArray listTx = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(this.X.Length);

                Array.Copy(this.X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = this.Tx[index].Data[0];
            }

            return new TestDataSet(listX, listTx);
        }
    }
}
