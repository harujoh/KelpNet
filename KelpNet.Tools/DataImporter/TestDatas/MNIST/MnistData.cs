namespace KelpNet.Tools
{
    public class MnistData
    {
        //訓練データ
        public LabeledDataSet Train;

        //評価データ
        public LabeledDataSet Eval;

        public MnistData()
        {
            MnistDataLoader mnistDataLoader = new MnistDataLoader();

            //訓練用データ
            Real[][] x = new Real[mnistDataLoader.TrainData.Length][];
            Real[] xLabel = new Real[mnistDataLoader.TrainData.Length];

            for (int i = 0; i < mnistDataLoader.TrainData.Length; i++)
            {
                x[i] = new Real[1 * 28 * 28];

                for (int j = 0; j < mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[i][j] = mnistDataLoader.TrainData[i][j] / 255.0;
                }

                xLabel[i] = mnistDataLoader.TrainLabel[i];
            }

            this.Train = new LabeledDataSet(x, xLabel, new[] { 1, 28, 28 });


            //評価用データ
            Real[][] y = new Real[mnistDataLoader.TeachData.Length][];
            Real[] yLabel = new Real[mnistDataLoader.TeachData.Length];

            for (int i = 0; i < mnistDataLoader.TeachData.Length; i++)
            {
                y[i] = new Real[1 * 28 * 28];

                for (int j = 0; j < mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[i][j] = mnistDataLoader.TeachData[i][j] / 255.0;
                }

                yLabel[i] = mnistDataLoader.TeachLabel[i];
            }

            this.Eval = new LabeledDataSet(y, yLabel, new[] { 1, 28, 28 });
        }
    }
}
