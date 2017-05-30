using System;
using RealType = System.Double;
//using RealType = System.Single;

namespace KelpNet.Common
{
    [Serializable]
    public struct Real:IComparable
    {
        public readonly RealType Value;

        private Real(RealType value)
        {
            this.Value = value;
        }

        public static implicit operator Real(RealType value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        public int CompareTo(object other)
        {
            return this.Value.CompareTo(((Real)other).Value);
        }

        public override string ToString()
        {
            return this.Value.ToString();
        }
    }
}
