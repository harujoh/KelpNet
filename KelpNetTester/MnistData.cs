using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using MNISTLoader;

namespace KelpNetTester
{
    class MnistData
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();

        private double[][,,] X;
        private int[][] Tx;

        private double[][,,] Y;
        private int[][] Ty;

        public MnistData()
        {
            //トレーニングデータ
            this.X = new double[this.mnistDataLoader.TrainData.Length][,,];
            //トレーニングデータラベル
            this.Tx = new int[this.mnistDataLoader.TrainData.Length][];

            for (int i = 0; i < this.mnistDataLoader.TrainData.Length; i++)
            {
                this.X[i] = new double[1, 28, 28];
                Buffer.BlockCopy(this.mnistDataLoader.TrainData[i].Select(val => val / 255.0).ToArray(), 0, this.X[i], 0, sizeof(double) *this.X[i].Length);
                this.Tx[i] = new [] { (int)this.mnistDataLoader.TrainLabel[i] };
            }

            //教師データ
            this.Y = new double[this.mnistDataLoader.TeachData.Length][,,];
            //教師データラベル
            this.Ty = new int[this.mnistDataLoader.TeachData.Length][];
            
            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                this.Y[i] = new double[1, 28, 28];
                Buffer.BlockCopy(this.mnistDataLoader.TeachData[i].Select(val => val / 255.0).ToArray(), 0, this.Y[i], 0, sizeof(double) *this.Y[i].Length);
                this.Ty[i] = new[] { (int)this.mnistDataLoader.TeachLabel[i] };
            }
        }

        public MnistDataSet GetRandomYSet(int dataCount)
        {
            List<double[,,]> listY = new List<double[,,]>();
            List<int[]> listTy = new List<int[]>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                listY.Add(this.Y[index]);
                listTy.Add(this.Ty[index]);
            }

            return new MnistDataSet(listY.ToArray(),listTy.ToArray());
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            List<double[,,]> listX = new List<double[,,]>();
            List<int[]> listTx = new List<int[]>();

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
        public Array[] Data;
        public int[][] Label;

        public MnistDataSet(Array[] data, int[][] label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
