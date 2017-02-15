#region License

/*

Copyright (c) 2009 - 2011 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Collections.Generic;
    using Cloo.Bindings;

    /// <summary>
    /// Represents an OpenCL 2D image.
    /// </summary>
    /// <seealso cref="ComputeImage"/>
    public class ComputeImage2D : ComputeImage
    {
        #region Constructors
        /*
        /// <summary>
        /// Creates a new <see cref="ComputeImage2D"/> from a <c>Bitmap</c>.
        /// </summary>
        /// <param name="context"> A valid <see cref="ComputeContext"/> in which the <see cref="ComputeImage2D"/> is created. </param>
        /// <param name="flags"> A bit-field that is used to specify allocation and usage information about the <see cref="ComputeImage2D"/>. </param>
        /// <param name="bitmap"> The bitmap to use. </param>
        /// <remarks> Note that only bitmaps with <c>Format32bppArgb</c> pixel format are currently supported. </remarks>
        public ComputeImage2D(ComputeContext context, ComputeMemoryFlags flags, Bitmap bitmap)
            :base(context, flags)
        {
            unsafe
            {                
                if(bitmap.PixelFormat != PixelFormat.Format32bppArgb)
                    throw new ArgumentException("Pixel format not supported.");
                
                //ComputeImageFormat format = Tools.ConvertImageFormat(bitmap.PixelFormat);
                ComputeImageFormat format = new ComputeImageFormat(ComputeImageChannelOrder.Bgra, ComputeImageChannelType.UnsignedInt8);
                BitmapData bitmapData = bitmap.LockBits(new Rectangle(new Point(), bitmap.Size), ImageLockMode.ReadOnly, bitmap.PixelFormat);

                try
                {
                    ComputeErrorCode error = ComputeErrorCode.Success;
                    Handle = CL10.CreateImage2D(
                        context.Handle,
                        flags,
                        &format,
                        new IntPtr(bitmap.Width),
                        new IntPtr(bitmap.Height),
                        new IntPtr(bitmapData.Stride),
                        bitmapData.Scan0,
                        &error);
                    ComputeException.ThrowOnError(error);
                }
                finally
                {
                    bitmap.UnlockBits(bitmapData);
                }

                Init();
            }
        }*/

        /// <summary>
        /// Creates a new <see cref="ComputeImage2D"/>.
        /// </summary>
        /// <param name="context"> A valid <see cref="ComputeContext"/> in which the <see cref="ComputeImage2D"/> is created. </param>
        /// <param name="flags"> A bit-field that is used to specify allocation and usage information about the <see cref="ComputeImage2D"/>. </param>
        /// <param name="format"> A structure that describes the format properties of the <see cref="ComputeImage2D"/>. </param>
        /// <param name="width"> The width of the <see cref="ComputeImage2D"/> in pixels. </param>
        /// <param name="height"> The height of the <see cref="ComputeImage2D"/> in pixels. </param>
        /// <param name="rowPitch"> The size in bytes of each row of elements of the <see cref="ComputeImage2D"/>. If <paramref name="rowPitch"/> is zero, OpenCL will compute the proper value based on <see cref="ComputeImage.Width"/> and <see cref="ComputeImage.ElementSize"/>. </param>
        /// <param name="data"> The data to initialize the <see cref="ComputeImage2D"/>. Can be <c>IntPtr.Zero</c>. </param>
        public ComputeImage2D(ComputeContext context, ComputeMemoryFlags flags, ComputeImageFormat format, int width, int height, long rowPitch, IntPtr data)
            : base(context, flags)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL10.CreateImage2D(context.Handle, flags, ref format, new IntPtr(width), new IntPtr(height), new IntPtr(rowPitch), data, out error);
            ComputeException.ThrowOnError(error);

            Init();
        }

        private ComputeImage2D(CLMemoryHandle handle, ComputeContext context, ComputeMemoryFlags flags)
            : base(context, flags)
        {
            Handle = handle;

            Init();
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Creates a new <see cref="ComputeImage2D"/> from an OpenGL renderbuffer object.
        /// </summary>
        /// <param name="context"> A <see cref="ComputeContext"/> with enabled CL/GL sharing. </param>
        /// <param name="flags"> A bit-field that is used to specify usage information about the <see cref="ComputeImage2D"/>. Only <c>ComputeMemoryFlags.ReadOnly</c>, <c>ComputeMemoryFlags.WriteOnly</c> and <c>ComputeMemoryFlags.ReadWrite</c> are allowed. </param>
        /// <param name="renderbufferId"> The OpenGL renderbuffer object id to use. </param>
        /// <returns> The created <see cref="ComputeImage2D"/>. </returns>
        public static ComputeImage2D CreateFromGLRenderbuffer(ComputeContext context, ComputeMemoryFlags flags, int renderbufferId)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            CLMemoryHandle image = CL10.CreateFromGLRenderbuffer(context.Handle, flags, renderbufferId, out error);
            ComputeException.ThrowOnError(error);

            return new ComputeImage2D(image, context, flags);
        }

        /// <summary>
        /// Creates a new <see cref="ComputeImage2D"/> from an OpenGL 2D texture object.
        /// </summary>
        /// <param name="context"> A <see cref="ComputeContext"/> with enabled CL/GL sharing. </param>
        /// <param name="flags"> A bit-field that is used to specify usage information about the <see cref="ComputeImage2D"/>. Only <c>ComputeMemoryFlags.ReadOnly</c>, <c>ComputeMemoryFlags.WriteOnly</c> and <c>ComputeMemoryFlags.ReadWrite</c> are allowed. </param>
        /// <param name="textureTarget"> One of the following values: GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, or GL_TEXTURE_RECTANGLE. Using GL_TEXTURE_RECTANGLE for texture_target requires OpenGL 3.1. Alternatively, GL_TEXTURE_RECTANGLE_ARB may be specified if the OpenGL extension GL_ARB_texture_rectangle is supported. </param>
        /// <param name="mipLevel"> The mipmap level of the OpenGL 2D texture object to be used. </param>
        /// <param name="textureId"> The OpenGL 2D texture object id to use. </param>
        /// <returns> The created <see cref="ComputeImage2D"/>. </returns>
        public static ComputeImage2D CreateFromGLTexture2D(ComputeContext context, ComputeMemoryFlags flags, int textureTarget, int mipLevel, int textureId)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            CLMemoryHandle image = CL10.CreateFromGLTexture2D(context.Handle, flags, textureTarget, mipLevel, textureId, out error);
            ComputeException.ThrowOnError(error);

            return new ComputeImage2D(image, context, flags);
        }

        /// <summary>
        /// Gets a collection of supported <see cref="ComputeImage2D"/> <see cref="ComputeImageFormat"/>s in a <see cref="ComputeContext"/>.
        /// </summary>
        /// <param name="context"> The <see cref="ComputeContext"/> for which the collection of <see cref="ComputeImageFormat"/>s is queried. </param>
        /// <param name="flags"> The <c>ComputeMemoryFlags</c> for which the collection of <see cref="ComputeImageFormat"/>s is queried. </param>
        /// <returns> The collection of the required <see cref="ComputeImageFormat"/>s. </returns>
        public static ICollection<ComputeImageFormat> GetSupportedFormats(ComputeContext context, ComputeMemoryFlags flags)
        {
            return GetSupportedFormats(context, flags, ComputeMemoryType.Image2D);
        }

        #endregion
    }
}