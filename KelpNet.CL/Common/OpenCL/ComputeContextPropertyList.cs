using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace KelpNet.CL.Common
{
    public class ComputeContextPropertyList: ICollection<ComputeContextProperty>
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private IList<ComputeContextProperty> properties;

        public ComputeContextPropertyList(ComputePlatform platform)
        {
            properties = new List<ComputeContextProperty>();
            properties.Add(new ComputeContextProperty(ComputeContextPropertyName.Platform, platform.handle));
        }

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

        public void Add(ComputeContextProperty item)
        {
            properties.Add(item);
        }

        public void Clear()
        {
            properties.Clear();
        }

        public bool Contains(ComputeContextProperty item)
        {
            return properties.Contains(item);
        }

        public void CopyTo(ComputeContextProperty[] array, int arrayIndex)
        {
            properties.CopyTo(array, arrayIndex);
        }

        public int Count
        {
            get { return properties.Count; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(ComputeContextProperty item)
        {
            return properties.Remove(item);
        }

        public IEnumerator<ComputeContextProperty> GetEnumerator()
        {
            return properties.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)properties).GetEnumerator();
        }
    }

    public class ComputeContextProperty
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContextPropertyName name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly IntPtr value;

        public ComputeContextPropertyName Name { get { return name; } }

        public IntPtr Value { get { return value; } }

        public ComputeContextProperty(ComputeContextPropertyName name, IntPtr value)
        {
            this.name = name;
            this.value = value;
        }
    }
}