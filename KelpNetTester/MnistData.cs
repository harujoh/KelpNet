using System.Collections.Generic;
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
            List<NdArray> listY = new List<NdArray>();
            List<NdArray> listTy = new List<NdArray>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                listY.Add(this.Y[index]);
                listTy.Add(this.Ty[index]);
            }

            return new MnistDataSet(listY.ToArray(), listTy.ToArray());
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            List<NdArray> listX = new List<NdArray>();
            List<NdArray> listTx = new List<NdArray>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.X.Length);

                listX.Add(this.X[index]);
                listTx.Add(this.Tx[index]);
            }

            return new MnistDataSet(listX.ToArray(), listTx.ToArray());
        }
    }

    public class MnistDataSet
    {
        public NdArray[] Data;
        public NdArray[] Label;

        public MnistDataSet(NdArray[] data, NdArray[] label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
