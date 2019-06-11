using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace KelpNet.Tools
{
    public class LabeledDataMaker
    {
        // TargetFolder
        // ├ CategoryA
        // │ ├ ImageA1.jpg
        // │ ├ ImageA2.jpg
        // │ └ ImageA3.jpg
        // ├ CategoryB
        // │ ├ ImageB1.jpg
        // │ ├ ImageB2.jpg
        // │ ├ ImageB3.jpg
        // │ └ ImageB4.jpg
        // └ CategoryC
        //    ├ ImageC1.jpg
        //    └ ImageC1.jpg

        public static LabeledDataSet MakeFromFolder(string foldersPath, int width = -1, int height = -1, bool eraseAlphaCh = true, bool makeValidData = false, bool makeTrainIndex = true)
        {
            List<Real[]> data = new List<Real[]>();
            List<Real> dataLabel = new List<Real>();
            List<string> labelName = new List<string>();

            string[] folders = Directory.GetDirectories(foldersPath);

            int bitcount = eraseAlphaCh ? 3 : -1;

            for (int i = 0; i < folders.Length; i++)
            {
                //クラス名称を保存
                labelName.Add(Path.GetFileName(folders[i]));

                string[] files = Directory.GetFiles(folders[i]);

                for (int j = 0; j < files.Length; j++)
                {
                    Bitmap baseBmp = new Bitmap(files[j]);

                    PixelFormat pixelFormat = baseBmp.PixelFormat;

                    //アルファチャンネルを削除する
                    if (eraseAlphaCh && Image.GetPixelFormatSize(baseBmp.PixelFormat) == 32)
                    {
                        pixelFormat = PixelFormat.Format24bppRgb;
                    }

                    if (width == -1)
                    {
                        width = baseBmp.Width;
                    }

                    if (height == -1)
                    {
                        height = baseBmp.Height;
                    }

                    //dataとdatalabelに水増ししつつ値をセット
                    SetAugmentatedBmp(data, dataLabel, baseBmp, width, height, pixelFormat, i);
                }
            }

            return new LabeledDataSet(data.ToArray(), dataLabel.ToArray(), new[] { bitcount, height, width }, labelName.ToArray(), makeValidData, makeTrainIndex, 9);
        }

        static void SetResizedBmp(List<Real[]> data, List<Real> label, Bitmap baseBmp, int width, int height, PixelFormat pixelFormat, int labelIndex)
        {
            Bitmap resultBmp = new Bitmap(width, height, pixelFormat);
            Graphics g = Graphics.FromImage(resultBmp);
            g.InterpolationMode = InterpolationMode.Bilinear;
            g.DrawImage(baseBmp, 0, 0, width, height);
            g.Dispose();

            data.Add(BitmapConverter.Image2RealArray(resultBmp));
            label.Add(labelIndex);
        }

        //SizeRatioは100分率で指定 10を指定した場合10%縮めた範囲を9回切り抜いて出力する
        static void SetAugmentatedBmp(List<Real[]> data, List<Real> label, Bitmap baseBmp, int width, int height, PixelFormat pixelFormat, int labelIndex, int sizeRatio = 10)
        {
            Bitmap resultBmp = new Bitmap(width, height, pixelFormat);
            Rectangle desRect = new Rectangle(0, 0, width, height);

            int widthOffset = baseBmp.Width / sizeRatio;
            int heightOffset = baseBmp.Height / sizeRatio;

            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    Rectangle srcRect = new Rectangle(x * widthOffset / 2, y * heightOffset / 2, baseBmp.Width - widthOffset, baseBmp.Height - heightOffset);

                    Graphics g = Graphics.FromImage(resultBmp);
                    g.InterpolationMode = InterpolationMode.Bilinear;
                    g.DrawImage(baseBmp, desRect, srcRect, GraphicsUnit.Pixel);
                    g.Dispose();

                    data.Add(BitmapConverter.Image2RealArray(resultBmp));
                    label.Add(labelIndex);
                }
            }
        }

        static void SetRotateImage(List<Real[]> data, List<Real> label, Bitmap baseBmp, int labelIndex)
        {
            //90
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray(baseBmp));
            label.Add(labelIndex);

            //180
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray(baseBmp));
            label.Add(labelIndex);

            //270
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray(baseBmp));
            label.Add(labelIndex);

            //もとに戻す
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
        }
    }
}
