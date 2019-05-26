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

        public static LabeledDataSet MakeFromFolder(string foldersPath, int width = -1, int height = -1, bool eraseAlphaCh = true)
        {
            List<LabeledData> data = new List<LabeledData>();
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
# if DEBUG
                    else
                    {
                        if (bitcount == -1)
                        {
                            bitcount = Image.GetPixelFormatSize(baseBmp.PixelFormat) / 8;
                        }
                        else
                        {
                            if(bitcount != Image.GetPixelFormatSize(baseBmp.PixelFormat) / 8) throw new Exception();
                        }
                    }
# endif

                    if (width == -1)
                    {
                        width = baseBmp.Width;
                    }
# if DEBUG
                    else
                    {
                        if(width != baseBmp.Width) throw new Exception();
                    }
# endif

                    if (height == -1)
                    {
                        height = baseBmp.Height;
                    }
# if DEBUG
                    else
                    {
                        if(height != baseBmp.Height) throw new Exception();
                    }
# endif

                    Bitmap resultBmp = new Bitmap(width, height, pixelFormat);
                    Graphics g = Graphics.FromImage(resultBmp);
                    g.InterpolationMode = InterpolationMode.Bilinear;
                    g.DrawImage(baseBmp, 0, 0, width, height);
                    g.Dispose();

                    data.Add(new LabeledData(BitmapConverter.Image2RealArray(resultBmp), i));
                }
            }

            return new LabeledDataSet(data.ToArray(), new[] { bitcount, height, width }, labelName.ToArray());
        }
    }
}
