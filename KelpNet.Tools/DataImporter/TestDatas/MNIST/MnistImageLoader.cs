using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

namespace KelpNet.Tools
{
    /// <summary>
    /// MNIST の画像をロードするためのクラス.
    /// http://yann.lecun.com/exdb/mnist/
    /// </summary>
    class MnistImageLoader
    {
        /// <summary>
        /// 0x0000 から始まるマジックナンバー.
        /// 0x00000803 (2051) が入る.
        /// </summary>
        public int magicNumber;

        /// <summary>
        /// 画像の数.
        /// </summary>
        public int numberOfImages;

        /// <summary>
        /// 画像の縦方向のサイズ.
        /// </summary>
        public int numberOfRows;

        /// <summary>
        /// 画像の横方向のサイズ.
        /// </summary>
        public int numberOfColumns;

        /// <summary>
        /// 画像の配列.
        /// Bitmap 形式で取得する場合は GetBitmap(index) を使用する.
        /// </summary>
        public List<byte[]> bitmapList;

        /// <summary>
        /// コンストラクタ.
        /// </summary>
        public MnistImageLoader()
        {
            this.bitmapList = new List<byte[]>();
        }

        /// <summary>
        /// MNIST のデータをロードする.
        /// 失敗した時は null を返す.
        /// </summary>
        /// <param name="path">画像データのパス.</param>
        /// <returns></returns>
        public static MnistImageLoader Load(string path)
        {
            // ファイルが存在しない
            if (File.Exists(path) == false)
            {
                return null;
            }

            MnistImageLoader loader = new MnistImageLoader();

            // バイト配列を分解する
            using (FileStream inStream = new FileStream(path, FileMode.Open, FileAccess.Read))
            using (GZipStream decompStream = new GZipStream(inStream, CompressionMode.Decompress))
            {
                BinaryReaderBE reader = new BinaryReaderBE(decompStream);

                loader.magicNumber = reader.ReadInt32();
                loader.numberOfImages = reader.ReadInt32();
                loader.numberOfRows = reader.ReadInt32();
                loader.numberOfColumns = reader.ReadInt32();

                int pixelCount = loader.numberOfRows * loader.numberOfColumns;
                for (int i = 0; i < loader.numberOfImages; i++)
                {
                    byte[] pixels = reader.ReadBytes(pixelCount);
                    loader.bitmapList.Add(pixels);
                }

                reader.Close();
            }

            return loader;
        }

        /// <summary>
        /// 引数で指定されたインデックス番号の画像を Bitmap 形式で取得する.
        /// 失敗した場合は null を返す.
        /// </summary>
        /// <param name="index">画像のインデックス番号.</param>
        /// <returns></returns>
        public Bitmap GetBitmap(int index)
        {
            // 範囲チェック
            if (index < 0 || index >= this.bitmapList.Count)
            {
                return null;
            }

            // Bitmap 画像を作成する
            Bitmap bitmap = new Bitmap(
                this.numberOfColumns,
                this.numberOfRows,
                PixelFormat.Format24bppRgb
            );
            BitmapData bitmapData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadWrite,
                bitmap.PixelFormat
            );


            byte[] pixels = this.bitmapList[index];
            IntPtr intPtr = bitmapData.Scan0;
            for (int y = 0; y < this.numberOfRows; y++)
            {
                int offsetY = bitmapData.Stride * y;
                for (int x = 0; x < this.numberOfColumns; x++)
                {
                    byte b = pixels[x + y * this.numberOfColumns];
                    // 次の行をコメントアウトすると白黒反転します
                    b = (byte)~b;
                    int offset = x * 3 + offsetY;
                    Marshal.WriteByte(intPtr, offset + 0, b);
                    Marshal.WriteByte(intPtr, offset + 1, b);
                    Marshal.WriteByte(intPtr, offset + 2, b);
                }
            }

            bitmap.UnlockBits(bitmapData);
            return bitmap;
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
            stringBuilder.AppendFormat("\tnumberOfImages: {0}", this.numberOfImages);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfRows: {0}", this.numberOfRows);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfColumns: {0}", this.numberOfColumns);
            return stringBuilder.ToString();
        }
    }
}

