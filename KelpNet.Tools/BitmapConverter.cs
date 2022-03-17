using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.Tools
{
#if !DOUBLE
    public class BitmapConverter
    {
        public static NdArray<T> Image2NdArray<T>(Bitmap input, bool isNorm = true, bool isToBgrArray = false, T[] bias = null) where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(float)) return (NdArray<T>)(object)BitmapConverterF.Image2NdArray(input, isNorm, isToBgrArray, (float[])(object)bias);
            if (typeof(T) == typeof(double)) return (NdArray<T>)(object)BitmapConverterD.Image2NdArray(input, isNorm, isToBgrArray, (double[])(object)bias);
            throw new Exception();
        }

        public static Bitmap[] NdArray2Image<T>(NdArray<T> input, bool isNorm = true, bool isFromBgrArray = false) where T : unmanaged, IComparable<T>
        {
            if (input is NdArray<float> inputF) return BitmapConverterF.NdArray2Image(inputF, isNorm, isFromBgrArray);
            if (input is NdArray<double> inputD) return BitmapConverterD.NdArray2Image(inputD, isNorm, isFromBgrArray);
            throw new Exception();
        }

        public static T[] Image2RealArray<T>(Bitmap input, bool isNorm = true, bool isToBgrArray = false, T[] bias = null) where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(float)) return (T[])(object)BitmapConverterF.Image2RealArray(input, isNorm, isToBgrArray);
            if (typeof(T) == typeof(double)) return (T[])(object)BitmapConverterD.Image2RealArray(input, isNorm, isToBgrArray);
            throw new Exception();
        }
    }

#endif

#if DOUBLE
    public static class BitmapConverterD
#else
    public static class BitmapConverterF
#endif
    {
        //Bitmapは [RGBRGB...]でデータが格納されているが多くの機械学習は[RR..GG..BB..]を前提にしているため入れ替えを行っている
        //Biasのチャンネル順は入力イメージに準ずる
        public static NdArray<Real> Image2NdArray(Bitmap input, bool isNorm = true, bool isToBgrArray = false, Real[] bias = null)
        {
            int bitcount = Image.GetPixelFormatSize(input.PixelFormat) / 8;

            if (bias == null || bitcount != bias.Length)
            {
                bias = new Real[bitcount];
            }

            Real norm = isNorm ? 255 : 1;

            NdArray<Real> result = new NdArray<Real>(bitcount, input.Height, input.Width);

            BitmapData bmpdat = input.LockBits(new Rectangle(0, 0, input.Width, input.Height), ImageLockMode.ReadOnly, input.PixelFormat);
            byte[] imageData = new byte[bmpdat.Stride * bmpdat.Height];

            Marshal.Copy(bmpdat.Scan0, imageData, 0, imageData.Length);

            input.UnlockBits(bmpdat);

            if (isToBgrArray)
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = bitcount - 1; ch >= 0; ch--)
                        {
                            result.Data[ch * input.Height * input.Width + y * input.Width + x] =
                                (imageData[y * bmpdat.Stride + x * bitcount + ch] + bias[ch]) / norm;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = 0; ch < bitcount; ch++)
                        {
                            result.Data[ch * input.Height * input.Width + y * input.Width + x] =
                                (imageData[y * bmpdat.Stride + x * bitcount + ch] + bias[ch]) / norm;
                        }
                    }
                }
            }

            return result;
        }

        public static Real[] Image2RealArray(Bitmap input, bool isNorm = true, bool isToBgrArray = false)
        {
            int bitcount = Image.GetPixelFormatSize(input.PixelFormat) / 8;

            Real norm = isNorm ? 255 : 1;

            BitmapData bmpdat = input.LockBits(new Rectangle(0, 0, input.Width, input.Height), ImageLockMode.ReadOnly, input.PixelFormat);
            byte[] imageData = new byte[bmpdat.Stride * bmpdat.Height];

            Marshal.Copy(bmpdat.Scan0, imageData, 0, imageData.Length);

            input.UnlockBits(bmpdat);

            Real[] result = new Real[bmpdat.Stride * input.Height];

            if (isToBgrArray)
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = bitcount - 1; ch >= 0; ch--)
                        {
                            result[ch * input.Height * input.Width + y * input.Width + x] = imageData[y * bmpdat.Stride + x * bitcount + ch] / norm;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = 0; ch < bitcount; ch++)
                        {
                            result[ch * input.Height * input.Width + y * input.Width + x] = imageData[y * bmpdat.Stride + x * bitcount + ch] / norm;
                        }
                    }
                }
            }

            return result;
        }

        public static Bitmap[] NdArray2Image(NdArray<Real> input, bool isNorm = true, bool isFromBgrArray = false)
        {
            Bitmap[] result = new Bitmap[input.BatchCount];

            for (int i = 0; i < result.Length; i++)
            {
                NdArray<Real> tmp = input.GetSingleArray(i);

                if (input.Shape.Length == 2)
                {
                    result[i] = CreateMonoImage(tmp.Data, input.Shape[0], input.Shape[1], isNorm);
                }
                else if (input.Shape.Length == 3)
                {
                    if (input.Shape[0] == 1)
                    {
                        result[i] = CreateMonoImage(tmp.Data, input.Shape[1], input.Shape[2], isNorm);
                    }
                    else if (input.Shape[0] == 3)
                    {
                        result[i] = CreateColorImage(tmp.Data, input.Shape[1], input.Shape[2], isNorm, isFromBgrArray);
                    }
                }
            }

            return result;
        }

        static Bitmap CreateMonoImage(Real[] data, int width, int height, bool isNorm)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            Real norm = isNorm ? 255 : 1;

            ColorPalette pal = result.Palette;
            for (int i = 0; i < 255; i++)
            {
                pal.Entries[i] = Color.FromArgb(i, i, i);
            }
            result.Palette = pal;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);

            byte[] resultData = new byte[bmpdat.Stride * height];

            if (isNorm)
            {
                Real datamax = data.Max();

                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x] = data[y * width + x] > 0
                            ? (byte)(data[y * width + x] / datamax * norm)
                            : (byte)0;
                    }
                }
            }
            else
            {
                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x] = data[y * width + x] > 0 ? (byte)data[y * width + x] : (byte)0;
                    }
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }

        static Bitmap CreateColorImage(Real[] data, int width, int height, bool isNorm, bool isFromBgrArray)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            Real norm = isNorm ? 255 : 1;
            int bitcount = Image.GetPixelFormatSize(result.PixelFormat) / 8;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);

            byte[] resultData = new byte[bmpdat.Stride * height];

            Real datamax = data.Max();

            if (isFromBgrArray)
            {
                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x * bitcount + 0] = (byte)(data[2 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 1] = (byte)(data[1 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 2] = (byte)(data[0 * height * width + y * width + x] / datamax * norm);
                    }
                }
            }
            else
            {
                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x * bitcount + 0] = (byte)(data[0 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 1] = (byte)(data[1 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 2] = (byte)(data[2 * height * width + y * width + x] / datamax * norm);
                    }
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }
    }
}
