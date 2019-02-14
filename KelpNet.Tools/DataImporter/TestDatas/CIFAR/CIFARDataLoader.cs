using System;
using System.Collections.Generic;
using System.Text;

namespace KelpNet.Tools
{
    public class CIFARDataLoader
    {
        const string DOWNLOAD_URL = "http://www.cs.toronto.edu/~kriz/";

        const string CIFAR10 = "cifar-10-binary.tar.gz";
        private readonly string[] CIFAR10TrainNames =
        {
            "cifar-10-batches-bin/data_batch_1.bin",
            "cifar-10-batches-bin/data_batch_2.bin",
            "cifar-10-batches-bin/data_batch_3.bin",
            "cifar-10-batches-bin/data_batch_4.bin",
            "cifar-10-batches-bin/data_batch_5.bin",
        };
        private readonly string CIFAR10TestName = "cifar-10-batches-bin/test_batch.bin";
        private const int CIFAR10_DATA_COUNT = 10000;

        const string CIFAR100 = "cifar-100-binary.tar.gz";
        private readonly string CIFAR100TrainName = "cifar-100-binary/train.bin";
        private readonly string CIFAR100TestName = "cifar-100-binary/test.bin";
        private const int CIFAR100_DATA_COUNT = 50000;
        private const int CIFAR100_TEST_DATA_COUNT = 10000;


        public string[] LabelNames;
        public string[] FineLabelNames;

        public byte[] TrainLabel;
        public byte[] TrainFineLabel;
        public byte[][] TrainData;

        public byte[] TestLabel;
        public byte[] TestFineLabel;
        public byte[][] TestData;

        private const int LABEL_SIZE = 1;
        private const int DATA_SIZE = 3072;

        public CIFARDataLoader(bool isCifar100 = false)
        {
            if (!isCifar100)
            {
                string cifar10Path = InternetFileDownloader.Donwload(DOWNLOAD_URL + CIFAR10, CIFAR10);
                Dictionary<string, byte[]> data = Tar.GetExtractedStreams(cifar10Path);

                this.LabelNames = Encoding.ASCII.GetString(data["cifar-10-batches-bin/batches.meta.txt"]).Split(new[] {'\n'}, StringSplitOptions.RemoveEmptyEntries);

                List<byte> trainLabel = new List<byte>();
                List<byte[]> trainData = new List<byte[]>();

                for (int i = 0; i < CIFAR10TrainNames.Length; i++)
                {
                    for (int j = 0; j < CIFAR10_DATA_COUNT; j++)
                    {
                        trainLabel.Add(data[CIFAR10TrainNames[i]][j * (DATA_SIZE + LABEL_SIZE)]);
                        byte[] tmpArray = new byte[DATA_SIZE];
                        Array.Copy(data[CIFAR10TrainNames[i]], j * (DATA_SIZE + LABEL_SIZE) + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                        trainData.Add(tmpArray);
                    }
                }

                this.TrainLabel = trainLabel.ToArray();
                this.TrainData = trainData.ToArray();

                List<byte> testLabel = new List<byte>();
                List<byte[]> testData = new List<byte[]>();

                for (int j = 0; j < CIFAR10_DATA_COUNT; j++)
                {
                    testLabel.Add(data[CIFAR10TestName][j * (DATA_SIZE + LABEL_SIZE)]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR10TestName], j * (DATA_SIZE + LABEL_SIZE) + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                    testData.Add(tmpArray);
                }

                this.TestLabel = testLabel.ToArray();
                this.TestData = testData.ToArray();
            }
            else
            {
                string cifar100Path = InternetFileDownloader.Donwload(DOWNLOAD_URL + CIFAR100, CIFAR100);
                Dictionary<string, byte[]> data = Tar.GetExtractedStreams(cifar100Path);

                //簡素なラベル名称
                this.LabelNames = Encoding.ASCII.GetString(data["cifar-100-binary/coarse_label_names.txt"]).Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
                //詳細なラベル名称
                this.FineLabelNames = Encoding.ASCII.GetString(data["cifar-100-binary/fine_label_names.txt"]).Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                List<byte> trainLabel = new List<byte>();
                List<byte> trainFineLabel = new List<byte>();
                List<byte[]> trainData = new List<byte[]>();

                for (int j = 0; j < CIFAR100_DATA_COUNT; j++)
                {
                    trainLabel.Add(data[CIFAR100TrainName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE)]);
                    trainFineLabel.Add(data[CIFAR100TrainName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR100TrainName], j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                    trainData.Add(tmpArray);
                }

                this.TrainLabel = trainLabel.ToArray();
                this.TrainFineLabel = trainFineLabel.ToArray();
                this.TrainData = trainData.ToArray();

                List<byte> testLabel = new List<byte>();
                List<byte> testFineLabel = new List<byte>();
                List<byte[]> testData = new List<byte[]>();

                for (int j = 0; j < CIFAR100_TEST_DATA_COUNT; j++)
                {
                    testLabel.Add(data[CIFAR100TestName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE)]);
                    testFineLabel.Add(data[CIFAR100TestName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR100TestName], j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                    testData.Add(tmpArray);
                }

                this.TestLabel = testLabel.ToArray();
                this.TestFineLabel = testFineLabel.ToArray();
                this.TestData = testData.ToArray();
            }
        }
    }
}
