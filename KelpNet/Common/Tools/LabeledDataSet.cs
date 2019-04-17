using System;

namespace KelpNet
{
    public class LabeledDataSet
    {
        public Real[][] Data;
        public Real[] Label;
        public int[] Shape;

        public LabeledDataSet(Real[][] data, int[] shape, Real[] label)
        {
            Data = data;
            Label = label;
            Shape = shape;
        }

        //データをランダムに取得しバッチにまとめる
        public TestDataSet GetRandomDataSet(int batchCount)
        {
            TestDataSet result = new TestDataSet(new NdArray(Shape, batchCount), new NdArray(new[] { 1 }, batchCount));

            for (int i = 0; i < batchCount; i++)
            {
                int index = Mother.Dice.Next(Label.Length);

                Array.Copy(Data[index], 0, result.Data.Data, i * result.Data.Length, result.Data.Length);
                result.Label.Data[i] = Label[index];
            }

            return result;
        }
    }
}
