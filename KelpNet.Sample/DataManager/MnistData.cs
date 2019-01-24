using System;
using KelpNet.Tools.DataImporter.TestDatas.MNIST;

namespace KelpNet.Sample.DataManager
{
    class MnistData<T> where T : unmanaged, IComparable<T>
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();

        private NdArray<T>[] X;
        private NdArray<T>[] Tx;

        private NdArray<T>[] Y;
        private NdArray<T>[] Ty;

        public MnistData()
        {
            //トレーニングデータ
            this.X = new NdArray<T>[this.mnistDataLoader.TrainData.Length];
            //トレーニングデータラベル
            this.Tx = new NdArray<T>[this.mnistDataLoader.TrainData.Length];

            for (int i = 0; i < this.mnistDataLoader.TrainData.Length; i++)
            {
                Real<T>[] x = new Real<T>[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = this.mnistDataLoader.TrainData[i][j] / 255.0f;
                }
                this.X[i] = new NdArray<T>(x, new[] { 1, 28, 28 });

                this.Tx[i] = new NdArray<T>(new[] { this.mnistDataLoader.TrainLabel[i] });
            }

            //教師データ
            this.Y = new NdArray<T>[this.mnistDataLoader.TeachData.Length];
            //教師データラベル
            this.Ty = new NdArray<T>[this.mnistDataLoader.TeachData.Length];

            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                Real<T>[] y = new Real<T>[28 * 28];
                for (int j = 0; j < this.mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[j] = this.mnistDataLoader.TeachData[i][j] / 255.0f;
                }
                this.Y[i] = new NdArray<T>(y, new[] { 1, 28, 28 });

                this.Ty[i] = new NdArray<T>(new[] { this.mnistDataLoader.TeachLabel[i] });
            }
        }

        public TestDataSet<T> GetRandomYSet(int dataCount)
        {
            NdArray<T> listY = new NdArray<T>(new[] { 1, 28, 28 }, dataCount);
            NdArray<T> listTy = new NdArray<T>(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother<T>.Dice.Next(this.Y.Length);

                Array.Copy(this.Y[index].Data, 0, listY.Data,i * listY.Length,listY.Length);
                listTy.Data[i] = this.Ty[index].Data[0];
            }

            return new TestDataSet<T>(listY, listTy);
        }

        public TestDataSet<T> GetRandomXSet(int dataCount)
        {
            NdArray<T> listX = new NdArray<T>(new[] { 1, 28, 28 }, dataCount);
            NdArray<T> listTx = new NdArray<T>(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother<T>.Dice.Next(this.X.Length);

                Array.Copy(this.X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = this.Tx[index].Data[0];
            }

            return new TestDataSet<T>(listX, listTx);
        }
    }
}
