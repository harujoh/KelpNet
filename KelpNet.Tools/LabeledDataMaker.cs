using System;
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

        public static LabeledDataSet<T> MakeFromFolder<T>(string foldersPath, int width = -1, int height = -1, bool eraseAlphaCh = true, bool makeValidData = false, bool makeTrainIndex = true) where T : unmanaged, IComparable<T>
        {
            List<T[]> data = new List<T[]>();
            List<int> dataLabel = new List<int>();
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

            return new LabeledDataSet<T>(data.ToArray(), dataLabel.ToArray(), new[] { bitcount, height, width }, labelName.ToArray(), makeValidData, makeTrainIndex, 9);
        }

        static void SetResizedBmp<T>(List<T[]> data, List<int> label, Bitmap baseBmp, int width, int height, PixelFormat pixelFormat, int labelIndex) where T : unmanaged, IComparable<T>
        {
            Bitmap resultBmp = new Bitmap(width, height, pixelFormat);
            Graphics g = Graphics.FromImage(resultBmp);
            g.InterpolationMode = InterpolationMode.Bilinear;
            g.DrawImage(baseBmp, 0, 0, width, height);
            g.Dispose();

            data.Add(BitmapConverter.Image2RealArray<T>(resultBmp));
            label.Add(labelIndex);
        }

        //SizeRatioは100分率で指定 10を指定した場合10%縮めた範囲を9回切り抜いて出力する
        public static void SetAugmentatedBmp<T>(List<T[]> data, List<int> label, Bitmap baseBmp, int width, int height, PixelFormat pixelFormat, int labelIndex, int sizeRatio = 10) where T : unmanaged, IComparable<T>
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

                    data.Add(BitmapConverter.Image2RealArray<T>(resultBmp));
                    label.Add(labelIndex);
                }
            }
        }

        static void SetRotateImage<T>(List<T[]> data, List<int> label, Bitmap baseBmp, int labelIndex) where T : unmanaged, IComparable<T>
        {
            //90
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray<T>(baseBmp));
            label.Add(labelIndex);

            //180
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray<T>(baseBmp));
            label.Add(labelIndex);

            //270
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
            data.Add(BitmapConverter.Image2RealArray<T>(baseBmp));
            label.Add(labelIndex);

            //もとに戻す
            baseBmp.RotateFlip(RotateFlipType.Rotate90FlipNone);
        }
    }
}
