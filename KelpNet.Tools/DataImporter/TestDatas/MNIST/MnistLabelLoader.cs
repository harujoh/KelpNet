using System.IO;
using System.IO.Compression;
using System.Text;

namespace KelpNet.Tools
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
            using (FileStream inStream = new FileStream(path, FileMode.Open, FileAccess.Read))
            using (GZipStream decompStream = new GZipStream(inStream, CompressionMode.Decompress))
            {
                // バイト配列を分解する
                BinaryReaderBE reader = new BinaryReaderBE(decompStream);

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
