using System;
using System.Collections.Generic;
using System.Linq;
using MNISTLoader;

namespace KelpNetTester
{
    class MnistData
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();
        readonly Random rnd = new Random();

        private double[][,,] X;
        private byte[][] Tx;

        private double[][,,] Y;
        private byte[][] Ty;

        public MnistData()
        {
            //トレーニングデータ
            X = new double[this.mnistDataLoader.TrainData.Length][,,];
            //トレーニングデータラベル
            Tx = new byte[this.mnistDataLoader.TrainData.Length][];

            for (int i = 0; i < this.mnistDataLoader.TrainData.Length; i++)
            {
                X[i] = new double[1, 28, 28];
                Buffer.BlockCopy(this.mnistDataLoader.TrainData[i].Select(val => val / 255.0).ToArray(), 0, X[i], 0, sizeof(double) * X[i].Length);
                Tx[i] = new[] { this.mnistDataLoader.TrainLabel[i] };
            }

            //教師データ
            Y = new double[this.mnistDataLoader.TeachData.Length][,,];
            //教師データラベル
            Ty = new byte[this.mnistDataLoader.TeachData.Length][];

            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                Y[i] = new double[1, 28, 28];
                Buffer.BlockCopy(this.mnistDataLoader.TeachData[i].Select(val => val / 255.0).ToArray(), 0, Y[i], 0, sizeof(double) * Y[i].Length);
                Ty[i] = new[] { this.mnistDataLoader.TeachLabel[i] };
            }

        }

        public MnistDataSet GetRandomYSet(int dataCount)
        {
            var listY = new List<double[,,]>();
            var listTy = new List<byte[]>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = rnd.Next(Y.Length);

                listY.Add(Y[index]);
                listTy.Add(Ty[index]);
            }

            return new MnistDataSet(listY.ToArray(),listTy.ToArray());
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            var listX = new List<double[,,]>();
            var listTx = new List<byte[]>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = rnd.Next(X.Length);

                listX.Add(X[index]);
                listTx.Add(Tx[index]);
            }

            return new MnistDataSet(listX.ToArray(), listTx.ToArray());
        }
    }

    public class MnistDataSet
    {
        public double[][,,] Data;
        public byte[][] Label;

        public MnistDataSet(double[][,,] data, byte[][] label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
