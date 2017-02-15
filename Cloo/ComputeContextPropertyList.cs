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
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;

    /// <summary>
    /// Represents a list of <see cref="ComputeContextProperty"/>s.
    /// </summary>
    /// <remarks> A <see cref="ComputeContextPropertyList"/> is used to specify the properties of a <see cref="ComputeContext"/>. </remarks>
    /// <seealso cref="ComputeContext"/>
    /// <seealso cref="ComputeContextProperty"/>
    public class ComputeContextPropertyList: ICollection<ComputeContextProperty>
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private IList<ComputeContextProperty> properties;

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new <see cref="ComputeContextPropertyList"/> which contains a single item specifying a <see cref="ComputePlatform"/>.
        /// </summary>
        /// <param name="platform"> A <see cref="ComputePlatform"/>. </param>
        public ComputeContextPropertyList(ComputePlatform platform)
        {
            properties = new List<ComputeContextProperty>();
            properties.Add(new ComputeContextProperty(ComputeContextPropertyName.Platform, platform.Handle.Value));
        }

        /// <summary>
        /// Creates a new <see cref="ComputeContextPropertyList"/> which contains the specified <see cref="ComputeContextProperty"/>s.
        /// </summary>
        /// <param name="properties"> An enumerable of <see cref="ComputeContextProperty"/>'s. </param>
        public ComputeContextPropertyList(IEnumerable<ComputeContextProperty> properties)
        {
            this.properties = new List<ComputeContextProperty>(properties);
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Gets a <see cref="ComputeContextProperty"/> of a specified <c>ComputeContextPropertyName</c>.
        /// </summary>
        /// <param name="name"> The <see cref="ComputeContextPropertyName"/> of the <see cref="ComputeContextProperty"/>. </param>
        /// <returns> The requested <see cref="ComputeContextProperty"/> or <c>null</c> if no such <see cref="ComputeContextProperty"/> exists in the <see cref="ComputeContextPropertyList"/>. </returns>
        public ComputeContextProperty GetByName(ComputeContextPropertyName name)
        {
            foreach (ComputeContextProperty property in properties)
                if (property.Name == name)
                    return property;

            return null;
        }

        #endregion

        #region Internal methods

        internal IntPtr[] ToIntPtrArray()
        {
            IntPtr[] result = new IntPtr[2 * properties.Count + 1];
            for (int i = 0; i < properties.Count; i++)
            {
                result[2 * i] = new IntPtr((int)properties[i].Name);
                result[2 * i + 1] = properties[i].Value;
            }
            result[result.Length - 1] = IntPtr.Zero;
            return result;
        }

        #endregion

        #region ICollection<ComputeContextProperty> Members

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        public void Add(ComputeContextProperty item)
        {
            properties.Add(item);
        }

        /// <summary>
        /// 
        /// </summary>
        public void Clear()
        {
            properties.Clear();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Contains(ComputeContextProperty item)
        {
            return properties.Contains(item);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex"></param>
        public void CopyTo(ComputeContextProperty[] array, int arrayIndex)
        {
            properties.CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// 
        /// </summary>
        public int Count
        {
            get { return properties.Count; }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Remove(ComputeContextProperty item)
        {
            return properties.Remove(item);
        }

        #endregion

        #region IEnumerable<ComputeContextProperty> Members

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public IEnumerator<ComputeContextProperty> GetEnumerator()
        {
            return ((IEnumerable<ComputeContextProperty>)properties).GetEnumerator();
        }

        #endregion

        #region IEnumerable Members

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)properties).GetEnumerator();
        }

        #endregion
    }
}