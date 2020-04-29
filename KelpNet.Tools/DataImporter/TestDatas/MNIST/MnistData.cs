using System;

namespace KelpNet.Tools
{
    //元データはbyte配列なのでこれをT配列に変換しつつノーマライズをかける
    public class MnistData<T> where T : unmanaged, IComparable<T>
    {
        //訓練データ
        public LabeledDataSet<T> Train;

        //評価データ
        public LabeledDataSet<T> Eval;

        public MnistData()
        {
            MnistDataLoader mnistDataLoader = new MnistDataLoader();

            this.Train = createLabeledDataSet(mnistDataLoader.TrainData, mnistDataLoader.TrainLabel);
            this.Eval = createLabeledDataSet(mnistDataLoader.TeachData, mnistDataLoader.TeachLabel);
        }

        LabeledDataSet<T> createLabeledDataSet(byte[][] data, byte[] label)
        {
            T[][] x = new T[data.Length][];
            int[] xLabel = new int[label.Length];

            //型を判定し画素を0.0～1.0にノーマライズ
            switch (x)
            {
                case float[][] xF:
                    for (int i = 0; i < data.Length; i++)
                    {
                        xF[i] = new float[1 * 28 * 28];

                        for (int j = 0; j < data[i].Length; j++)
                        {
                            xF[i][j] = data[i][j] / 255.0f;
                        }
                    }
                    break;

                case double[][] xD:
                    for (int i = 0; i < data.Length; i++)
                    {
                        xD[i] = new double[1 * 28 * 28];

                        for (int j = 0; j < data[i].Length; j++)
                        {
                            xD[i][j] = data[i][j] / 255.0;
                        }
                    }
                    break;
            }

            Array.Copy(label,xLabel,label.Length);
            return new LabeledDataSet<T>(x, xLabel, new[] { 1, 28, 28 });
        }
    }
}
