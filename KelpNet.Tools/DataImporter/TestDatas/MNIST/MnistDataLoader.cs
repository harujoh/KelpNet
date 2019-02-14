namespace KelpNet.Tools
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

        const string TRAIN_LABEL_HASH = "d53e105ee54ea40749a09fcbcd1e9432";
        const string TRAIN_IMAGE_HASH = "f68b3c2dcbeaaa9fbdd348bbdeb94873";
        const string TEACH_LABEL_HASH = "ec29112dd5afa0611ce80d1b7f02629c";
        const string TEACH_IMAGE_HASH = "9fb629c4189551a2d022fa330f9573f3";

        public MnistDataLoader()
        {
            string trainlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_LABEL, TRAIN_LABEL, TRAIN_LABEL_HASH);
            MnistLabelLoader trainLabelLoader = MnistLabelLoader.Load(trainlabelPath);
            this.TrainLabel = trainLabelLoader.labelList;

            string trainimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_IMAGE, TRAIN_IMAGE, TRAIN_IMAGE_HASH);
            MnistImageLoader trainImageLoader = MnistImageLoader.Load(trainimagePath);
            this.TrainData = trainImageLoader.bitmapList.ToArray();


            string teachlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_LABEL, TEACH_LABEL, TEACH_LABEL_HASH);
            MnistLabelLoader teachLabelLoader = MnistLabelLoader.Load(teachlabelPath);
            this.TeachLabel = teachLabelLoader.labelList;

            string teachimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_IMAGE, TEACH_IMAGE, TEACH_IMAGE_HASH);
            MnistImageLoader teachImageLoader = MnistImageLoader.Load(teachimagePath);
            this.TeachData = teachImageLoader.bitmapList.ToArray();
        }
    }
}
