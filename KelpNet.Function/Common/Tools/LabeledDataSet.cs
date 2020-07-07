using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;

namespace KelpNet
{
    public class LabeledDataSet<T> where T : unmanaged, IComparable<T>
    {
        public static Type[] KnownTypes =
        {
            typeof(T[]),
            typeof(int[]),
            typeof(string[]),
        };

        DataContractSerializer bf = new DataContractSerializer(typeof(object), KnownTypes);
        private bool[] Loaded;

        private T[][] Data;
        private int[] DataLabel;

        private int[] trainIndex;
        private int[] validIndex;
        private int[] testIndex;

        public int Length = 0;
        public string[] LabelNames;
        public int[] Shape;

        public string FileName;

        public LabeledDataSet(T[][] data, int[] dataLabel, int[] shape, string[] labelNames = null, bool makeValidData = false, bool makeTrainIndex = true, int augmentCount = 1)
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
                int maxLabel = 0;

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
            this.FileName = fileName;

            using (ZipArchive zipArchiveData = ZipFile.OpenRead(fileName))
            {
                ZipArchiveEntry zipLength = zipArchiveData.GetEntry("Length");
                Length = (int)bf.ReadObject(zipLength.Open());

                ZipArchiveEntry zipShape = zipArchiveData.GetEntry("Shape");
                Shape = (int[])bf.ReadObject(zipShape.Open());

                ZipArchiveEntry zipLabelName = zipArchiveData.GetEntry("LabelNames");
                LabelNames = (string[])bf.ReadObject(zipLabelName.Open());

                ZipArchiveEntry zipLabel = zipArchiveData.GetEntry("DataLabel");
                DataLabel = (int[])bf.ReadObject(zipLabel.Open());

                if (isAllLoad)
                {
                    //全読み込み
                    Data = new T[Length][];

                    for (int i = 0; i < Data.Length; i++)
                    {
                        ZipArchiveEntry zipData = zipArchiveData.GetEntry(i.ToString());
                        Data[i] = (T[])bf.ReadObject(zipData.Open());
                    }

                    Loaded = Enumerable.Repeat(true, Length).ToArray();
                }
                else
                {
                    //遅延読込
                    Data = new T[Length][];
                    Loaded = new bool[Length];
                }

                if (makeTrainIndex)
                {
                    MakeTrainData(makeValidData);
                }
            }
        }

        public T[] Get(int i)
        {
            if (!Loaded[i])
            {
                using (ZipArchive zipArchiveData = ZipFile.OpenRead(this.FileName))
                {
                    ZipArchiveEntry zipData = zipArchiveData.GetEntry(i.ToString());
                    Data[i] = (T[])bf.ReadObject(zipData.Open());
                    Loaded[i] = true;
                }
            }

            return Data[i];
        }

        public T[] GetTrain(int i)
        {
            return Get(trainIndex[i]);
        }

        public T[] GetValid(int i)
        {
            return Get(validIndex[i]);
        }

        public T[] GetTest(int i)
        {
            return Get(testIndex[i]);
        }

        public void AllLoad()
        {
            using (ZipArchive zipArchiveData = ZipFile.OpenRead(this.FileName))
            {
                for (int i = 0; i < Loaded.Length; i++)
                {
                    if (!Loaded[i])
                    {
                        ZipArchiveEntry zipData = zipArchiveData.GetEntry(i.ToString());
                        Data[i] = (T[])bf.ReadObject(zipData.Open());
                        Loaded[i] = true;
                    }
                }
            }
        }

        //データをランダムに取得しバッチにまとめる
        public TestDataSet<T> GetRandomDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => Mother.Dice.Next(Data.Length));
        }

        public TestDataSet<T> GetRandomTrainDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => trainIndex[Mother.Dice.Next(trainIndex.Length)]);
        }

        public TestDataSet<T> GetRandomValidDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => validIndex[Mother.Dice.Next(validIndex.Length)]);
        }

        public TestDataSet<T> GetRandomTestDataSet(int batchCount)
        {
            return GetRandomData(batchCount, () => testIndex[Mother.Dice.Next(testIndex.Length)]);
        }

        private TestDataSet<T> GetRandomData(int batchCount, Func<int> getIndexFunc)
        {
            T[] data = new T[NdArray.ShapeToLength(Shape) * batchCount];
            int[] label = new int[batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                int index = getIndexFunc();

                T[] labeledData = Get(index);
                Array.Copy(labeledData, 0, data, i * labeledData.Length, labeledData.Length);

                label[i] = DataLabel[index];
            }

            TestDataSet<T> result = new TestDataSet<T>(NdArray.Convert(data, Shape, batchCount), NdArray.Convert(label, new[] { 1 }, batchCount));

            return result;
        }

        //全データを読み込む
        public TestDataSet<T> GetAllDataSet()
        {
            int batchCount = Data.Length;

            T[] data = new T[NdArray.ShapeToLength(Shape) * Data.Length];

            for (int i = 0; i < batchCount; i++)
            {
                T[] labeledData = Get(i);
                Array.Copy(labeledData, 0, data, i * labeledData.Length, labeledData.Length);
            }

            TestDataSet<T> result = new TestDataSet<T>(NdArray.Convert(data, Shape, batchCount), NdArray.Convert(DataLabel, new[] { 1 }, batchCount));

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
                        bf.WriteObject(stream, Data[i]);
                    }
                }

                ZipArchiveEntry zipLabelName = zipArchive.CreateEntry("LabelNames");
                using (Stream stream = zipLabelName.Open())
                {
                    bf.WriteObject(stream, LabelNames);
                }

                ZipArchiveEntry zipShape = zipArchive.CreateEntry("Shape");
                using (Stream stream = zipShape.Open())
                {
                    bf.WriteObject(stream, Shape);
                }

                ZipArchiveEntry zipLength = zipArchive.CreateEntry("Length");
                using (Stream stream = zipLength.Open())
                {
                    bf.WriteObject(stream, Length);
                }

                ZipArchiveEntry entryLabel = zipArchive.CreateEntry("DataLabel");
                using (Stream stream = entryLabel.Open())
                {
                    bf.WriteObject(stream, DataLabel);
                }
            }
        }

        public static LabeledDataSet<T> Load(string fileName, bool isAllLoad = false, bool makeValidData = false, bool makeTrainIndex = true)
        {
            return new LabeledDataSet<T>(fileName, isAllLoad, makeValidData, makeTrainIndex);
        }
    }
}
