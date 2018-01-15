namespace KelpNet.DataImporter.TestDatas.MNIST
{
    public class MnistDataLoader
    {
        public byte[] TrainLabel;
        public byte[][] TrainData;

        public byte[] TeachLabel;
        public byte[][] TeachData;

        const string DOWNLOAD_URL = "http://yann.lecun.com/exdb/mnist/";

        const string TRAIN_LABEL = "train-labels-idx1-ubyte.gz";
        const string TRAIN_IMAGE = "train-images-idx3-ubyte.gz";
        const string TEACH_LABEL = "t10k-labels-idx1-ubyte.gz";
        const string TEACH_IMAGE = "t10k-images-idx3-ubyte.gz";

        public MnistDataLoader()
        {
            string trainlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_LABEL, TRAIN_LABEL);
            MnistLabelLoader trainLabelLoader = MnistLabelLoader.Load(trainlabelPath);
            this.TrainLabel = trainLabelLoader.labelList;

            string trainimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_IMAGE, TRAIN_IMAGE);
            MnistImageLoader trainImageLoader = MnistImageLoader.Load(trainimagePath);
            this.TrainData = trainImageLoader.bitmapList.ToArray();


            string teachlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_LABEL, TEACH_LABEL);
            MnistLabelLoader teachLabelLoader = MnistLabelLoader.Load(teachlabelPath);
            this.TeachLabel = teachLabelLoader.labelList;

            string teachimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_IMAGE, TEACH_IMAGE);
            MnistImageLoader teachImageLoader = MnistImageLoader.Load(teachimagePath);
            this.TeachData = teachImageLoader.bitmapList.ToArray();
        }
    }
}
