using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;

namespace KelpNet
{
    public class LabeledDataSet
    {
        NetDataContractSerializer bf = new NetDataContractSerializer();
        private ZipArchive ZipArchiveData;
        private bool[] Loaded;
        private LabeledData[] Data;

        private int[] trainIndex;
        private int[] validIndex;
        private int[] testIndex;

        public int Length = 0;
        public string[] LabelName;
        public int[] Shape;

        public int ClassCount
        {
            get { return LabelName.Length; }
        }

        public LabeledData this[int i]
        {
            get
            {
                return Get(i);
            }
        }

        public LabeledDataSet(Real[][] data, int[] shape, Real[] label, string[] labelName = null, bool makeValidData = false, bool makeTrainIndex = true) :
            this(
                LabeledData.Convert(data, label),
                shape,
                labelName,
                makeValidData
            )
        {
        }

        public LabeledDataSet(LabeledData[] data, int[] shape, string[] labelName = null, bool makeValidData = false, bool makeTrainIndex = true)
        {
            Data = data;
            Loaded = Enumerable.Repeat(true, data.Length).ToArray();
            LabelName = labelName;
            Shape = shape;
            Length = data.Length;

            //ラベル名称が未指定の場合、連番を自動的に割り当てる
            if (LabelName == null)
            {
                Real maxLabel = 0;

                for (int i = 0; i < Data.Length; i++)
                {
                    if (Data[i].Label > maxLabel) maxLabel = Data[i].Label;
                }

                LabelName = Enumerable.Range(0, (int)maxLabel).Select(s => s.ToString()).ToArray();
            }

            if (makeTrainIndex)
            {
                MakeTrainData(makeValidData);
            }
        }

        void MakeTrainData(bool makeValidData)
        {
            //Train,Valid,Testデータを作成する

            //全データのインデックスを最初に持っておく
            List<int> trainIndexList = new List<int>(Enumerable.Range(0, Data.Length));

            //各ラベルで10％をテストデータに割り当てる
            int testDataCount = Data.Length / LabelName.Length / 10;

            //少なすぎる場合は一つ
            if (testDataCount == 0) testDataCount = 1;

            testIndex = GetRondomIndex(testDataCount * 10, trainIndexList);

            if (makeValidData)
            {
                validIndex = GetRondomIndex(testDataCount * 10, trainIndexList);
            }

            //残りをトレーニングデータとして使用する
            trainIndex = trainIndexList.ToArray();
        }

        private int[] GetRondomIndex(int dataCount, List<int> trainIndexList)
        {
            int[] result = new int[dataCount];

            for (int i = 0; i < result.Length; i++)
            {
                int randomIndex = -1;

                do
                {
                    randomIndex = trainIndexList[Mother.Dice.Next(trainIndexList.Count)];
                } while ((int)Get(randomIndex).Label != i % LabelName.Length);

                //全データのインデックスから使用分を除く
                trainIndexList.Remove(randomIndex);
                result[i] = randomIndex;
            }

            return result;
        }

        public LabeledDataSet(ZipArchive zipArchive, bool isAllLoad = false, bool makeValidData = false, bool makeTrainIndex = true)
        {
            ZipArchiveData = zipArchive;

            ZipArchiveEntry zipLength = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension("Length"));
            Length = (int)bf.Deserialize(zipLength.Open());

            ZipArchiveEntry zipShape = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension("Shape"));
            Shape = (int[])bf.Deserialize(zipShape.Open());

            ZipArchiveEntry zipLabelName = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension("LabelName"));
            LabelName = (string[])bf.Deserialize(zipLabelName.Open());

            if (isAllLoad)
            {
                //全読み込み
                Data = new LabeledData[Length];

                for (int i = 0; i < Data.Length; i++)
                {
                    ZipArchiveEntry zipData = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension(i.ToString()));
                    Data[i] = (LabeledData)bf.Deserialize(zipData.Open());
                }

                Loaded = Enumerable.Repeat(true, Length).ToArray();
            }
            else
            {
                //遅延読込
                Data = new LabeledData[Length];
                Loaded = new bool[Length];
            }

            if (makeTrainIndex)
            {
                MakeTrainData(makeValidData);
            }
        }

        public LabeledData Get(int i)
        {
            if (!Loaded[i])
            {
                ZipArchiveEntry zipData = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension(i.ToString()));
                Data[i] = (LabeledData)bf.Deserialize(zipData.Open());
                Loaded[i] = true;
            }

            return Data[i];
        }

        public LabeledData GetTrain(int i)
        {
            return Get(trainIndex[i]);
        }

        public LabeledData GetValid(int i)
        {
            return Get(validIndex[i]);
        }

        public LabeledData GetTest(int i)
        {
            return Get(testIndex[i]);
        }

        public void AllLoad()
        {
            for (int i = 0; i < Loaded.Length; i++)
            {
                if (!Loaded[i])
                {
                    ZipArchiveEntry zipData = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension(i.ToString()));
                    Data[i] = (LabeledData)bf.Deserialize(zipData.Open());
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

                LabeledData labeledData = Get(index);

                Array.Copy(labeledData.Data, 0, result.Data.Data, i * result.Data.Length, result.Data.Length);
                result.Label.Data[i] = labeledData.Label;
            }

            return result;
        }

        public void Save(string fileName)
        {
            //ZIP書庫を作成
            if (File.Exists(fileName))
            {
                File.Delete(fileName);
            }

            AllLoad();

            using (ZipArchive zipArchive = ZipFile.Open(fileName, ZipArchiveMode.Create))
            {
                for (int i = 0; i < Data.Length; i++)
                {
                    ZipArchiveEntry entry = zipArchive.CreateEntry(i.ToString());
                    using (Stream stream = entry.Open())
                    {
                        bf.Serialize(stream, Data[i]);
                    }
                }

                ZipArchiveEntry zipLabelName = zipArchive.CreateEntry("LabelName");
                using (Stream stream = zipLabelName.Open())
                {
                    bf.Serialize(stream, LabelName);
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
            }
        }

        public static LabeledDataSet Load(string fileName, bool isAllLoad = false, bool makeValidData = false, bool makeTrainIndex = true)
        {
            return new LabeledDataSet(ZipFile.OpenRead(fileName), isAllLoad, makeValidData, makeTrainIndex);
        }
    }
}
