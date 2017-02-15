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
    ///A structure of two integers of platform specific size.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SysIntX2
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

        #endregion

        #region Constructors

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public SysIntX2(int x, int y)
            : this(new IntPtr(x), new IntPtr(y))
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public SysIntX2(long x, long y)
            : this(new IntPtr(x), new IntPtr(y))
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public SysIntX2(IntPtr x, IntPtr y)
        {
            X = x;
            Y = y;
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Gets the string representation of the SysIntX2.
        /// </summary>
        /// <returns> The string representation of the SysIntX2. </returns>
        public override string ToString()
        {
            return X + " " + Y;
        }

        #endregion
    }
}