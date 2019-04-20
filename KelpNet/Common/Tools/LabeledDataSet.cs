using System;
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

        public int Length = 0;
        public string[] LabelName;
        public int[] Shape;

        public LabeledData this[int i]
        {
            get
            {
                return Get(i);
            }
        }

        public LabeledDataSet(Real[][] data, int[] shape, Real[] label, string[] labelName = null)
        {
            Data = LabeledData.Convert(data, label);
            Loaded = Enumerable.Repeat(true, data.Length).ToArray();
            LabelName = labelName;
            Shape = shape;
            Length = data.Length;
        }

        public LabeledDataSet(LabeledData[] data, int[] shape, string[] labelName = null)
        {
            Data = data;
            Loaded = Enumerable.Repeat(true, data.Length).ToArray();
            LabelName = labelName;
            Shape = shape;
            Length = data.Length;
        }

        public LabeledDataSet(ZipArchive zipArchive, bool isAllLoad = false)
        {
            ZipArchiveData = zipArchive;

            ZipArchiveEntry zipLength = ZipArchiveData.GetEntry(Path.GetFileNameWithoutExtension("Length"));
            Length = (int) bf.Deserialize(zipLength.Open());

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
            TestDataSet result = new TestDataSet(new NdArray(Shape, batchCount), new NdArray(new[] { 1 }, batchCount));

            for (int i = 0; i < batchCount; i++)
            {
                int index = Mother.Dice.Next(Data.Length);

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

        public static LabeledDataSet Load(string fileName, bool isAllLoad = false)
        {
            return new LabeledDataSet(ZipFile.OpenRead(fileName), isAllLoad);
        }
    }
}
