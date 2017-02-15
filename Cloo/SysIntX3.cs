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
    using System.Runtime.InteropServices;

    /// <summary>
    /// A structure of three integers of platform specific size.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SysIntX3
    {
        #region Fields

        /// <summary>
        /// The first coordinate.
        /// </summary>
        public IntPtr X;

        /// <summary>
        /// The second coordinate.
        /// </summary>
        public IntPtr Y;

        /// <summary>
        /// The third coordinate.
        /// </summary>
        public IntPtr Z;

        #endregion

        #region Constructors

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x2"></param>
        /// <param name="z"></param>
        public SysIntX3(SysIntX2 x2, long z)
            : this(x2.X, x2.Y, new IntPtr(z))
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public SysIntX3(int x, int y, int z)
            : this(new IntPtr(x), new IntPtr(y), new IntPtr(z))
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public SysIntX3(long x, long y, long z)
            : this(new IntPtr(x), new IntPtr(y), new IntPtr(z))
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public SysIntX3(IntPtr x, IntPtr y, IntPtr z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Gets the string representation of the SysIntX2.
        /// </summary>
        /// <returns> The string representation of the SysIntX2. </returns>
        public override string ToString()
        {
            return X + " " + Y + " " + Z;
        }

        #endregion
    }
}