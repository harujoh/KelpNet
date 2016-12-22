using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;

namespace KelpNet.Common.Tools
{
    public class NdArrayConverter
    {
        public static Bitmap NdArray2Image(NdArray input)
        {
            Bitmap result = new Bitmap(1, 1);

            if (input.Shape.Length == 2)
            {
                result = CreateMono(input.Data, input.Shape[0], input.Shape[1]);
            }
            else if (input.Shape.Length == 3)
            {
                if (input.Shape[0] == 1)
                {
                    result = CreateMono(input.Data, input.Shape[1], input.Shape[2]);
                }
                else if (input.Shape[0] == 3)
                {
                    result = new Bitmap(input.Shape[1], input.Shape[2], PixelFormat.Format24bppRgb);
                }
            }

            return result;
        }

        static Bitmap CreateMono(double[] data, int width, int height)
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
    }

}
