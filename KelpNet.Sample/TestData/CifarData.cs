using System;
using KelpNet.Common;
using KelpNet.Tools.DataImporter.TestDatas.CIFAR;

namespace KelpNet.Sample.TestData
{
    class CifarData
    {
        CIFARDataLoader cifarDataLoader = new CIFARDataLoader();

        private NdArray[] X;
        private NdArray[] Tx;

        private NdArray[] Y;
        private NdArray[] Ty;

        public readonly int ClassCount;


        public CifarData(bool isCifar100 = false, bool isFineLabel = false)
        {
            //トレーニングデータ
            this.X = new NdArray[this.cifarDataLoader.TrainData.Length];
            //トレーニングデータラベル
            this.Tx = new NdArray[this.cifarDataLoader.TrainData.Length];

            //Cifar100のときは100クラス、簡素であれば20クラス、Cifar10のときは10クラス分類
            ClassCount = isCifar100 ? isFineLabel ? 100 : 20 : 10;


            for (int i = 0; i < this.cifarDataLoader.TrainData.Length; i++)
            {
                Real[] x = new Real[3 * 32 * 32];

                for (int j = 0; j < this.cifarDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = this.cifarDataLoader.TrainData[i][j] / 255.0;
                }

                this.X[i] = new NdArray(x, new[] { 3, 32, 32 });

                if (isCifar100 & isFineLabel)
                {
                    this.Tx[i] = new NdArray(new[] { (Real)this.cifarDataLoader.TrainFineLabel[i] });
                }
                else
                {
                    this.Tx[i] = new NdArray(new[] { (Real)this.cifarDataLoader.TrainLabel[i] });
                }
            }

            //教師データ
            this.Y = new NdArray[this.cifarDataLoader.TestData.Length];
            //教師データラベル
            this.Ty = new NdArray[this.cifarDataLoader.TestData.Length];

            for (int i = 0; i < this.cifarDataLoader.TestData.Length; i++)
            {
                Real[] y = new Real[3 * 32 * 32];

                for (int j = 0; j < this.cifarDataLoader.TestData[i].Length; j++)
                {
                    y[j] = this.cifarDataLoader.TestData[i][j] / 255.0;
                }

                this.Y[i] = new NdArray(y, new[] { 3, 32, 32 });

                if (isCifar100 & isFineLabel)
                {
                    this.Ty[i] = new NdArray(new[] { (Real)this.cifarDataLoader.TestFineLabel[i] });
                }
                else
                {
                    this.Ty[i] = new NdArray(new[] { (Real)this.cifarDataLoader.TestLabel[i] });
                }
            }
        }

        public TestDataSet GetRandomYSet(int dataCount)
        {
            NdArray listY = new NdArray(new[] { 3, 32, 32 }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                Array.Copy(this.Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = this.Ty[index].Data[0];
            }

            return new TestDataSet(listY, listTy);
        }

        public TestDataSet GetRandomXSet(int dataCount)
        {
            NdArray listX = new NdArray(new[] { 3, 32, 32 }, dataCount);
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
