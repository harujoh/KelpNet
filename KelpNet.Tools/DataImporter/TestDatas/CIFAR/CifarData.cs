namespace KelpNet.Tools
{
    public class CifarData
    {
        //訓練データ
        public LabeledDataSet Train;

        //評価データ
        public LabeledDataSet Eval;

        public readonly int ClassCount;

        public CifarData(bool isCifar100 = false, bool isFineLabel = false)
        {
            CIFARDataLoader cifarDataLoader = new CIFARDataLoader(isCifar100);

            //訓練用データ
            Real[][] x = new Real[cifarDataLoader.TrainData.Length][];
            Real[] xLabel = new Real[cifarDataLoader.TrainData.Length];

            //Cifar100のときは100クラス、簡素であれば20クラス、Cifar10のときは10クラス分類
            ClassCount = isCifar100 ? isFineLabel ? 100 : 20 : 10;

            for (int i = 0; i < cifarDataLoader.TrainData.Length; i++)
            {
                x[i] = new Real[3 * 32 * 32];
                for (int j = 0; j < cifarDataLoader.TrainData[i].Length; j++)
                {
                    x[i][j] = cifarDataLoader.TrainData[i][j] / 255.0;
                }

                if (isCifar100 & isFineLabel)
                {

                    xLabel[i] = cifarDataLoader.TrainFineLabel[i];
                }
                else
                {
                    xLabel[i] = cifarDataLoader.TrainLabel[i];
                }
            }

            this.Train = new LabeledDataSet(x, xLabel, new[] { 3, 32, 32 });

            //評価用データ
            Real[][] y = new Real[cifarDataLoader.TestData.Length][];
            Real[] yLabel = new Real[cifarDataLoader.TestData.Length];

            for (int i = 0; i < cifarDataLoader.TestData.Length; i++)
            {
                y[i] = new Real[3 * 32 * 32];
                for (int j = 0; j < cifarDataLoader.TestData[i].Length; j++)
                {
                    y[i][j] = cifarDataLoader.TestData[i][j] / 255.0;
                }

                if (isCifar100 & isFineLabel)
                {
                    yLabel[i] = cifarDataLoader.TestFineLabel[i];
                }
                else
                {
                    yLabel[i] = cifarDataLoader.TestLabel[i];
                }
            }

            this.Eval = new LabeledDataSet(y, yLabel, new[] { 3, 32, 32 });
        }
    }
}
