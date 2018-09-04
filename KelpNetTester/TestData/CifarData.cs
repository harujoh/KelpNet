using System;
using CIFARLoader;
using KelpNet.Common;

namespace KelpNetTester.TestData
{
    class CifarData
    {
        private NdArray[] X;
        private NdArray[] Tx;
        private NdArray[] TxFine;

        private NdArray[] Y;
        private NdArray[] Ty;
        private NdArray[] TyFine;

        public CifarData(bool isCifar100 = false)
        {
            CIFARDataLoader cifarDataLoader = new CIFARDataLoader(isCifar100);

            //トレーニングデータ
            this.X = new NdArray[cifarDataLoader.TrainData.Length];
            //トレーニングデータラベル
            this.Tx = new NdArray[cifarDataLoader.TrainData.Length];
            //トレーニングデータラベル（詳細）
            this.TxFine = new NdArray[cifarDataLoader.TrainData.Length];

            for (int i = 0; i < cifarDataLoader.TrainData.Length; i++)
            {
                Real[] x = new Real[3 * 32 * 32];
                for (int j = 0; j < cifarDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = cifarDataLoader.TrainData[i][j] / 255.0;
                }
                this.X[i] = new NdArray(x, new[] { 3, 32, 32 });

                this.Tx[i] = new NdArray(new[] { (Real)cifarDataLoader.TrainLabel[i] });

                if (isCifar100)
                {
                    this.TxFine[i] = new NdArray(new[] {(Real) cifarDataLoader.TrainFineLabel[i]});
                }
                else
                {
                    //詳細ラベルはCifar100のみなのでダミーデータを挿入
                    this.TxFine[i] = new NdArray(new[] { (Real)cifarDataLoader.TrainLabel[i] });
                }
            }

            //教師データ
            this.Y = new NdArray[cifarDataLoader.TestData.Length];
            //教師データラベル
            this.Ty = new NdArray[cifarDataLoader.TestData.Length];
            //教師データラベル（詳細）
            this.TyFine = new NdArray[cifarDataLoader.TestData.Length];

            for (int i = 0; i < cifarDataLoader.TestData.Length; i++)
            {
                Real[] y = new Real[3 * 32 * 32];

                for (int j = 0; j < cifarDataLoader.TestData[i].Length; j++)
                {
                    y[j] = cifarDataLoader.TestData[i][j] / 255.0;
                }

                this.Y[i] = new NdArray(y, new[] { 3, 32, 32 });

                this.Ty[i] = new NdArray(new[] { (Real)cifarDataLoader.TestLabel[i] });

                if (isCifar100)
                {
                    this.TyFine[i] = new NdArray(new[] { (Real)cifarDataLoader.TestFineLabel[i] });
                }
                else
                {
                    //詳細ラベルはCifar100のみなのでダミーデータを挿入
                    this.TyFine[i] = new NdArray(new[] { (Real)cifarDataLoader.TrainLabel[i] });
                }
            }
        }

        //トレーニングデータを取得
        public TestDataSet GetRandomYSet(int dataCount, bool isFineLabel = false)
        {
            NdArray listY = new NdArray(new[] { 3, 32, 32 }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                Array.Copy(this.Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = isFineLabel ? this.TyFine[index].Data[0] : this.Ty[index].Data[0];
            }

            return new TestDataSet(listY, listTy);
        }

        //教師データを取得
        public TestDataSet GetRandomXSet(int dataCount, bool isFineLabel = false)
        {
            NdArray listX = new NdArray(new[] { 3, 32, 32 }, dataCount);
            NdArray listTx = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(this.X.Length);

                Array.Copy(this.X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = isFineLabel ? this.TxFine[index].Data[0] : this.Tx[index].Data[0];
            }

            return new TestDataSet(listX, listTx);
        }
    }
}
