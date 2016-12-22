using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;

namespace KelpNet.Common.Tools
{
    public class NdArrayConverter
    {
        //Bitmapは [RGBRGB...]でデータが格納されているが多くの機械学習は[RR..GG..BB..]を前提にしているため入れ替えを行っている
        public static NdArray Image2NdArray(Bitmap input)
        {
            int bitcount = Image.GetPixelFormatSize(input.PixelFormat) / 8;
            NdArray result = NdArray.Zeros(bitcount, input.Height, input.Width);

            BitmapData bmpdat = input.LockBits(new Rectangle(0, 0, input.Width, input.Height), ImageLockMode.ReadOnly, input.PixelFormat);
            byte[] imageData = new byte[bmpdat.Stride * bmpdat.Height];

            Marshal.Copy(bmpdat.Scan0, imageData, 0, imageData.Length);

            for (int y = 0; y < input.Height; y++)
            {
                for (int x = 0; x < input.Width; x++)
                {
                    for (int ch = 0; ch < bitcount; ch++)
                    {
                        result.Data[ch * input.Height * input.Width + y * input.Width + x] =
                            imageData[y * bmpdat.Stride + x * bitcount + ch] / 255.0;
                    }
                }
            }

            return result;
        }

        public static Bitmap NdArray2Image(NdArray input)
        {
            if (input.Shape.Length == 2)
            {
                return CreateMonoImage(input.Data, input.Shape[0], input.Shape[1]);
            }
            else if (input.Shape.Length == 3)
            {
                if (input.Shape[0] == 1)
                {
                    return CreateMonoImage(input.Data, input.Shape[1], input.Shape[2]);
                }
                else if (input.Shape[0] == 3)
                {
                    return CreateColorImage(input.Data, input.Shape[1], input.Shape[2]);
                }
            }

            return null;
        }

        static Bitmap CreateMonoImage(double[] data, int width, int height)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format8bppIndexed);

            ColorPalette pal = result.Palette;
            for (int i = 0; i < 255; i++)
            {
                pal.Entries[i] = Color.FromArgb(i, i, i);
            }
            result.Palette = pal;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);

            byte[] resultData = new byte[bmpdat.Stride * height];

            double datamax = data.Max();

            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < result.Width; x++)
                {
                    resultData[y * bmpdat.Stride + x] = (byte)(data[y * width + x] / datamax * 255);
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }

        static Bitmap CreateColorImage(double[] data, int width, int height)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            int bitcount = Image.GetPixelFormatSize(result.PixelFormat) / 8;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);

            byte[] resultData = new byte[bmpdat.Stride * height];

            double datamax = data.Max();

            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < result.Width; x++)
                {
                    resultData[y * bmpdat.Stride + x * bitcount + 0] = (byte)(data[0 * height * width + y * width + x] / datamax * 255);
                    resultData[y * bmpdat.Stride + x * bitcount + 1] = (byte)(data[1 * height * width + y * width + x] / datamax * 255);
                    resultData[y * bmpdat.Stride + x * bitcount + 2] = (byte)(data[2 * height * width + y * width + x] / datamax * 255);
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }

    }

}
