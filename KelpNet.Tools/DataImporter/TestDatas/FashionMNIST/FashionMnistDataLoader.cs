namespace KelpNet.Tools
{
    public class FashionMnistDataLoader
    {
        public byte[] TrainLabel;
        public byte[][] TrainData;

        public byte[] TeachLabel;
        public byte[][] TeachData;

        const string DOWNLOAD_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";

        const string TRAIN_LABEL = "train-labels-idx1-ubyte.gz";
        const string TRAIN_IMAGE = "train-images-idx3-ubyte.gz";
        const string TEACH_LABEL = "t10k-labels-idx1-ubyte.gz";
        const string TEACH_IMAGE = "t10k-images-idx3-ubyte.gz";

        const string TRAIN_LABEL_HASH = "25c81989df183df01b3e8a0aad5dffbe";
        const string TRAIN_IMAGE_HASH = "8d4fb7e6c68d591d4c3dfef9ec88bf0d";
        const string TEACH_LABEL_HASH = "bb300cfdad3c16e7a12a480ee83cd310";
        const string TEACH_IMAGE_HASH = "bef4ecab320f06d8554ea6380940ec79";

        public FashionMnistDataLoader()
        {
            //ファイル名がMNISTと同じ為、保存名は"fashion-"を足している
            string trainlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_LABEL, "fashion-" + TRAIN_LABEL, TRAIN_LABEL_HASH);
            MnistLabelLoader trainLabelLoader = MnistLabelLoader.Load(trainlabelPath);
            this.TrainLabel = trainLabelLoader.labelList;

            string trainimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_IMAGE, "fashion-" + TRAIN_IMAGE, TRAIN_IMAGE_HASH);
            MnistImageLoader trainImageLoader = MnistImageLoader.Load(trainimagePath);
            this.TrainData = trainImageLoader.bitmapList.ToArray();


            string teachlabelPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_LABEL, "fashion-" + TEACH_LABEL, TEACH_LABEL_HASH);
            MnistLabelLoader teachLabelLoader = MnistLabelLoader.Load(teachlabelPath);
            this.TeachLabel = teachLabelLoader.labelList;

            string teachimagePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEACH_IMAGE, "fashion-" + TEACH_IMAGE, TEACH_IMAGE_HASH);
            MnistImageLoader teachImageLoader = MnistImageLoader.Load(teachimagePath);
            this.TeachData = teachImageLoader.bitmapList.ToArray();
        }
    }
}
