using System;
using KelpNet.Tools.DataImporter.TestDatas.CIFAR;

namespace KelpNet.Sample.DataManager
{
    class CifarData<T> where T : unmanaged, IComparable<T>
    {
        CIFARDataLoader cifarDataLoader = new CIFARDataLoader();

        private NdArray<T>[] X;
        private NdArray<T>[] Tx;

        private NdArray<T>[] Y;
        private NdArray<T>[] Ty;

        public readonly int ClassCount;


        public CifarData(bool isCifar100 = false, bool isFineLabel = false)
        {
            //トレーニングデータ
            this.X = new NdArray<T>[this.cifarDataLoader.TrainData.Length];
            //トレーニングデータラベル
            this.Tx = new NdArray<T>[this.cifarDataLoader.TrainData.Length];

            //Cifar100のときは100クラス、簡素であれば20クラス、Cifar10のときは10クラス分類
            ClassCount = isCifar100 ? isFineLabel ? 100 : 20 : 10;


            for (int i = 0; i < this.cifarDataLoader.TrainData.Length; i++)
            {
                Real<T>[] x = new Real<T>[3 * 32 * 32];

                for (int j = 0; j < this.cifarDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = this.cifarDataLoader.TrainData[i][j] / 255.0f;
                }

                this.X[i] = new NdArray<T>(x, new[] { 3, 32, 32 });

                if (isCifar100 & isFineLabel)
                {
                    this.Tx[i] = new NdArray<T>(new[] { this.cifarDataLoader.TrainFineLabel[i] });
                }
                else
                {
                    this.Tx[i] = new NdArray<T>(new[] { this.cifarDataLoader.TrainLabel[i] });
                }
            }

            //教師データ
            this.Y = new NdArray<T>[this.cifarDataLoader.TestData.Length];
            //教師データラベル
            this.Ty = new NdArray<T>[this.cifarDataLoader.TestData.Length];

            for (int i = 0; i < this.cifarDataLoader.TestData.Length; i++)
            {
                Real<T>[] y = new Real<T>[3 * 32 * 32];

                for (int j = 0; j < this.cifarDataLoader.TestData[i].Length; j++)
                {
                    y[j] = this.cifarDataLoader.TestData[i][j] / 255.0f;
                }

                this.Y[i] = new NdArray<T>(y, new[] { 3, 32, 32 });

                if (isCifar100 & isFineLabel)
                {
                    this.Ty[i] = new NdArray<T>(new[] { this.cifarDataLoader.TestFineLabel[i] });
                }
                else
                {
                    this.Ty[i] = new NdArray<T>(new[] { this.cifarDataLoader.TestLabel[i] });
                }
            }
        }

        public TestDataSet<T> GetRandomYSet(int dataCount)
        {
            NdArray<T> listY = new NdArray<T>(new[] { 3, 32, 32 }, dataCount);
            NdArray<T> listTy = new NdArray<T>(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother<T>.Dice.Next(this.Y.Length);

                Array.Copy(this.Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = this.Ty[index].Data[0];
            }

            return new TestDataSet<T>(listY, listTy);
        }

        public TestDataSet<T> GetRandomXSet(int dataCount)
        {
            NdArray<T> listX = new NdArray<T>(new[] { 3, 32, 32 }, dataCount);
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
