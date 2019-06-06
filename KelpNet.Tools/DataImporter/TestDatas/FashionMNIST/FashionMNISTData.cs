namespace KelpNet.Tools
{
    public class FashionMnistData
    {
        //訓練データ
        public LabeledDataSet Train;

        //評価データ
        public LabeledDataSet Eval;

        public FashionMnistData()
        {
            FashionMnistDataLoader fashionMnistDataLoader = new FashionMnistDataLoader();

            //訓練用データ
            Real[][] x = new Real[fashionMnistDataLoader.TrainData.Length][];
            Real[] xLabel = new Real[fashionMnistDataLoader.TrainData.Length];

            for (int i = 0; i < fashionMnistDataLoader.TrainData.Length; i++)
            {
                x[i] = new Real[1 * 28 * 28];

                for (int j = 0; j < fashionMnistDataLoader.TrainData[i].Length; j++)
                {
                    x[i][j] = fashionMnistDataLoader.TrainData[i][j] / 255.0;
                }

                xLabel[i] = fashionMnistDataLoader.TrainLabel[i];
            }

            this.Train = new LabeledDataSet(x, xLabel, new[] { 1, 28, 28 });


            //評価用データ
            Real[][] y = new Real[fashionMnistDataLoader.TeachData.Length][];
            Real[] yLabel = new Real[fashionMnistDataLoader.TeachData.Length];

            for (int i = 0; i < fashionMnistDataLoader.TeachData.Length; i++)
            {
                y[i] = new Real[1 * 28 * 28];

                for (int j = 0; j < fashionMnistDataLoader.TeachData[i].Length; j++)
                {
                    y[i][j] = fashionMnistDataLoader.TeachData[i][j] / 255.0;
                }

                yLabel[i] = fashionMnistDataLoader.TeachLabel[i];
            }

            this.Eval = new LabeledDataSet(y, yLabel, new[] { 1, 28, 28 });
        }

    }
}
