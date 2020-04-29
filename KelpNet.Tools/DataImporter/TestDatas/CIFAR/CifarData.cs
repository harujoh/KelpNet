using System;

namespace KelpNet.Tools
{
    public class CifarData<T> where T : unmanaged, IComparable<T>
    {
        //訓練データ
        public LabeledDataSet<T> Train;

        //評価データ
        public LabeledDataSet<T> Eval;

        public readonly int ClassCount;

        public CifarData(bool isCifar100 = false, bool isFineLabel = false)
        {
            CIFARDataLoader cifarDataLoader = new CIFARDataLoader(isCifar100);

            //Cifar100のときは100クラス、簡素であれば20クラス、Cifar10のときは10クラス分類
            ClassCount = isCifar100 ? isFineLabel ? 100 : 20 : 10;

            if (isCifar100 & isFineLabel)
            {
                this.Train = createLabeledDataSet(cifarDataLoader.TrainData, cifarDataLoader.TrainFineLabel);
                this.Eval = createLabeledDataSet(cifarDataLoader.TestData, cifarDataLoader.TestFineLabel);
            }
            else
            {
                this.Train = createLabeledDataSet(cifarDataLoader.TrainData, cifarDataLoader.TrainLabel);
                this.Eval = createLabeledDataSet(cifarDataLoader.TestData, cifarDataLoader.TestLabel);
            }
        }

        LabeledDataSet<T> createLabeledDataSet(byte[][] data, byte[] label)
        {
            //訓練用データ
            T[][] x = new T[data.Length][];
            int[] xLabel = new int[label.Length];

            //型を判定し画素を0.0～1.0にノーマライズ
            switch (x)
            {
                case float[][] xF:
                    for (int i = 0; i < data.Length; i++)
                    {
                        xF[i] = new float[3 * 32 * 32];

                        for (int j = 0; j < data[i].Length; j++)
                        {
                            xF[i][j] = data[i][j] / 255.0f;
                        }
                    }
                    break;

                case double[][] xD:
                    for (int i = 0; i < data.Length; i++)
                    {
                        xD[i] = new double[3 * 32 * 32];

                        for (int j = 0; j < data[i].Length; j++)
                        {
                            xD[i][j] = data[i][j] / 255.0;
                        }
                    }
                    break;
            }

            Array.Copy(label, xLabel, label.Length);
            return new LabeledDataSet<T>(x, xLabel, new[] { 3, 32, 32 });
        }

    }
}
