using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

namespace KelpNet
{
    public class LabeledDataSet
    {
        NetDataContractSerializer bf = new NetDataContractSerializer();
        private ZipArchive ZipArchiveData;
        private bool[] Loaded;

        private Real[][] Data;
        private Real[] DataLabel;

        private int[] trainIndex;
        private int[] validIndex;
        private int[] testIndex;

        public int Length = 0;
        public string[] LabelNames;
        public int[] Shape;

        public LabeledDataSet(Real[][] data, Real[] dataLabel, int[] shape, string[] labelNames = null, bool makeValidData = false, bool makeTrainIndex = true, int augmentCount = 1)
        {
            Data = data;
            DataLabel = dataLabel;
            Loaded = Enumerable.Repeat(true, data.Length).ToArray();
            LabelNames = labelNames;
            Shape = shape;
            Length = data.Length;

            //ラベル名称が未指定の場合、連番を自動的に割り当てる
            if (LabelNames == null)
            {
                Real maxLabel = 0;

                for (int i = 0; i < DataLabel.Length; i++)
                {
                    if (DataLabel[i] > maxLabel) maxLabel = DataLabel[i];
                }

                LabelNames = Enumerable.Range(0, (int)maxLabel + 1).Select(s => s.ToString()).ToArray();
            }

            if (makeTrainIndex)
            {
                MakeTrainData(makeValidData, augmentCount);
            }
        }

        //Train,Valid,Testデータを作成する
        void MakeTrainData(bool makeValidData, int augmentCount = 1)
        {
            //全データのインデックスを最初に持っておく
            List<int> trainIndexList = new List<int>(Enumerable.Range(0, DataLabel.Length));

            //各クラスで水増し前の10％のデータをテストデータに割り当てる
            int testDataCount = DataLabel.Length / LabelNames.Length / 10 / augmentCount;

            //少なすぎる場合は一つ
            if (testDataCount == 0) testDataCount = 1;

            //テストデータ用のIndexを作成
            testIndex = GetRondomIndex(testDataCount, LabelNames.Length, augmentCount, trainIndexList);

            //Validはオプション
            if (makeValidData)
            {
                //検証データ用のIndexを作成
                validIndex = GetRondomIndex(testDataCount, LabelNames.Length, augmentCount, trainIndexList);
            }

            //残りをトレーニングデータとして使用する
            trainIndex = trainIndexList.ToArray();
        }

        //dataCount:１クラスに必要な数
        private int[] GetRondomIndex(int dataCount, int labelNameLength, int augmentCount, List<int> trainIndexList)
        {
            int[] result = new int[dataCount * labelNameLength * augmentCount];

            //各クラスのクジをどれだけ引いたかを管理
            int[] classDataCount = Enumerable.Repeat(0, labelNameLength).ToArray();

            for (int i = 0; i < dataCount * labelNameLength; i++)
            {
                int randomIndex = -1;

                //ほしいクラスが出るまでくじ引きを行う
                do
                {
                    randomIndex = trainIndexList[Mother.Dice.Next(trainIndexList.Count) / augmentCount * augmentCount];
                } while (classDataCount[(int)DataLabel[randomIndex]] == dataCount);

                classDataCount[(int)DataLabel[randomIndex]]++;

                for (int j = 0; j < augmentCount; j++)
                {
                    //全データのインデックスから使用分を除く
                    trainIndexList.Remove(randomIndex + j);
                    result[i * augmentCount + j] = randomIndex + j;
                }
            }

            return result;
        }

        private LabeledDataSet(string fileName, bool isAllLoad = false, bool makeValidData = false, bool makeTrainIndex = true)
        {
            ZipArchiveData = ZipFile.OpenRead(fileName);

            ZipArchiveEntry zipLength = ZipArchiveData.GetEntry("Length");
            Length = (int)bf.Deserialize(zipLength.Open());

            ZipArchiveEntry zipShape = ZipArchiveData.GetEntry("Shape");
            Shape = (int[])bf.Deserialize(zipShape.Open());

            ZipArchiveEntry zipLabelName = ZipArchiveData.GetEntry("LabelNames");
            LabelNames = (string[])bf.Deserialize(zipLabelName.Open());

            ZipArchiveEntry zipLabel = ZipArchiveData.GetEntry("DataLabel");
            DataLabel = (Real[])bf.Deserialize(zipLabel.Open());

            if (isAllLoad)
            {
                //全読み込み
                Data = new Real[Length][];

                for (int i = 0; i < Data.Length; i++)
                {
                    ZipArchiveEntry zipData = ZipArchiveData.GetEntry(i.ToString());
                    Data[i] = (Real[])bf.Deserialize(zipData.Open());
                }

                Loaded = Enumerable.Repeat(true, Length).ToArray();
            }
            else
            {
                //遅延読込
                Data = new Real[Length][];
                Loaded = new bool[Length];
            }

            if (makeTrainIndex)
            {
                MakeTrainData(makeValidData);
            }
        }

        public Real[] Get(int i)
        {
            if (!Loaded[i])
            {
                ZipArchiveEntry zipData = ZipArchiveData.GetEntry(i.ToString());
                Data[i] = (Real[])bf.Deserialize(zipData.Open());
                Loaded[i] = true;
            }

            return Data[i];
        }

        public Real[] GetTrain(int i)
        {
            return Get(trainIndex[i]);
        }

        public Real[] GetValid(int i)
        {
            return Get(validIndex[i]);
        }

        public Real[] GetTest(int i)
        {
            return Get(testIndex[i]);
        }

        public void AllLoad()
        {
            for (int i = 0; i < Loaded.Length; i++)
            {
                if (!Loaded[i])
                {
                    ZipArchiveEntry zipData = ZipArchiveData.GetEntry(i.ToString());
                    Data[i] = (Real[])bf.Deserialize(zipData.Open());
                    Loaded[i] = true;
                }
            }
        }

        //データをランダムに取得しバッチにまとめる
        public TestDataSet GetRandomDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => Mother.Dice.Next(Data.Length));
        }

        public TestDataSet GetRandomTrainDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => trainIndex[Mother.Dice.Next(trainIndex.Length)]);
        }

        public TestDataSet GetRandomValidDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => validIndex[Mother.Dice.Next(validIndex.Length)]);
        }

        public TestDataSet GetRandomTestDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => testIndex[Mother.Dice.Next(testIndex.Length)]);
        }

        private TestDataSet GetRandomData(int batchCount, Func<int> getIndexFunc)
        {
            TestDataSet result = new TestDataSet(new NdArray(Shape, batchCount), new NdArray(new[] { 1 }, batchCount));

            for (int i = 0; i < batchCount; i++)
            {
                int index = getIndexFunc();

                Real[] labeledData = Get(index);
                Array.Copy(labeledData, 0, result.Data.Data, i * result.Data.Length, result.Data.Length);

                result.Label.Data[i] = DataLabel[index];
            }

            return result;
        }

        public void Save(string savePath)
        {
            //ZIP書庫を作成
            if (File.Exists(savePath))
            {
                File.Delete(savePath);
            }

            AllLoad();

            using (ZipArchive zipArchive = ZipFile.Open(savePath, ZipArchiveMode.Create))
            {
                for (int i = 0; i < Data.Length; i++)
                {
                    ZipArchiveEntry entry = zipArchive.CreateEntry(i.ToString());
                    using (Stream stream = entry.Open())
                    {
                        bf.Serialize(stream, Data[i]);
                    }
                }

                ZipArchiveEntry zipLabelName = zipArchive.CreateEntry("LabelNames");
                using (Stream stream = zipLabelName.Open())
                {
                    bf.Serialize(stream, LabelNames);
                }

                ZipArchiveEntry zipShape = zipArchive.CreateEntry("Shape");
                using (Stream stream = zipShape.Open())
                {
                    bf.Serialize(stream, Shape);
                }

                ZipArchiveEntry zipLength = zipArchive.CreateEntry("Length");
                using (Stream stream = zipLength.Open())
                {
                    bf.Serialize(stream, Length);
                }

                ZipArchiveEntry entryLabel = zipArchive.CreateEntry("DataLabel");
                using (Stream stream = entryLabel.Open())
                {
                    bf.Serialize(stream, DataLabel);
                }
            }

            string classFileName = Path.Combine(Path.GetDirectoryName(savePath) ?? string.Empty, Path.GetFileNameWithoutExtension(savePath) + "Classes.txt");
            File.WriteAllLines(classFileName, LabelNames, Encoding.UTF8);
        }

        public static LabeledDataSet Load(string fileName, bool isAllLoad = false, bool makeValidData = false, bool makeTrainIndex = true)
        {
            return new LabeledDataSet(fileName, isAllLoad, makeValidData, makeTrainIndex);
        }
    }
}
