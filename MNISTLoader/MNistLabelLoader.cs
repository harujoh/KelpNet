using System.IO;
using System.Text;

namespace MNISTLoader
{
    /// <summary>
    /// MNIST のラベルファイルをロードするためのクラス.
    /// http://yann.lecun.com/exdb/mnist/
    /// </summary>
    class MnistLabelLoader
    {
        /// <summary>
        /// 0x0000 から始まるマジックナンバー.
        /// 0x00000801 (2049) が入る.
        /// </summary>
        public int magicNumber;

        /// <summary>
        /// ラベルの数.
        /// </summary>
        public int numberOfItems;

        /// <summary>
        /// ラベルの配列.
        /// </summary>
        public byte[] labelList;

        /// <summary>
        /// コンストラクタ.
        /// </summary>
        public MnistLabelLoader()
        {
        }

        /// <summary>
        /// MNIST のラベルファイルをロードする.
        /// 失敗した時は null を返す.
        /// </summary>
        /// <param name="path">ラベルファイルのパス.</param>
        /// <returns></returns>
        public static MnistLabelLoader Load(string path)
        {
            // ファイルが存在しない
            if (File.Exists(path) == false)
            {
                return null;
            }

            MnistLabelLoader loader = new MnistLabelLoader();

            // バイト配列を分解する
            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                BinaryReaderBE reader = new BinaryReaderBE(stream);

                loader.magicNumber = reader.ReadInt32();
                loader.numberOfItems = reader.ReadInt32();
                loader.labelList = reader.ReadBytes(loader.numberOfItems);

                reader.Close();
            }

            return loader;
        }

        /// <summary>
        /// デバッグ用.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(GetType().Name);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tmagicNumber: 0x{0:X8}", this.magicNumber);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfItems: {0}", this.numberOfItems);
            return stringBuilder.ToString();
        }
    }
}
