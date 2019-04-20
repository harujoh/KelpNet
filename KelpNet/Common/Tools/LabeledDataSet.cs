using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;

namespace KelpNet
{
    [Serializable]
    public class LabeledDataSet
    {
        public List<Real[]> Data = new List<Real[]>();
        public List<Real> Label = new List<Real>();
        public List<string> LabelName = new List<string>();
        public int[] Shape;

        public LabeledDataSet(Real[][] data, int[] shape, Real[] label, string[] labelName = null)
        {
            Data.AddRange(data);
            Label.AddRange(label);
            if (labelName != null) LabelName.AddRange(labelName);
            Shape = shape;
        }

        public LabeledDataSet(int[] shape)
        {
            Shape = shape;
        }

        public void Add(Real[] data, Real label)
        {
            Data.Add(data);
            Label.Add(label);
        }

        public void AddRange(Real[][] data, Real[] label)
        {
            Data.AddRange(data);
            Label.AddRange(label);
        }

        //データをランダムに取得しバッチにまとめる
        public TestDataSet GetRandomDataSet(int batchCount)
        {
            TestDataSet result = new TestDataSet(new NdArray(Shape, batchCount), new NdArray(new[] { 1 }, batchCount));

            for (int i = 0; i < batchCount; i++)
            {
                int index = Mother.Dice.Next(Label.Count);

                Array.Copy(Data[index], 0, result.Data.Data, i * result.Data.Length, result.Data.Length);
                result.Label.Data[i] = Label[index];
            }

            return result;
        }

        public void Save(string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();

            //ZIP書庫を作成
            using (ZipArchive zipArchive = ZipFile.Open(fileName, ZipArchiveMode.Create))
            {
                ZipArchiveEntry entry = zipArchive.CreateEntry(Path.GetFileNameWithoutExtension(fileName));

                using (Stream stream = entry.Open())
                {
                    // エントリにバイナリを書き込む
                    bf.Serialize(stream, this);
                }
            }
        }

        public static LabeledDataSet Load(string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();
            LabeledDataSet result;

            using (ZipArchive a = ZipFile.OpenRead(fileName))
            {
                ZipArchiveEntry e = a.GetEntry(Path.GetFileNameWithoutExtension(fileName));

                if (e != null)
                {
                    result = (LabeledDataSet)bf.Deserialize(e.Open());
                }
                else
                {
                    throw new Exception("ファイル形式が異なります");
                }
            }

            return result;
        }
    }
}
