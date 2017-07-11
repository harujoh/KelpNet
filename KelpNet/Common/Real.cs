using System;
//using RealType = System.Double;
using RealType = System.Single;

namespace KelpNet.Common
{
    [Serializable]
    public struct Real : IComparable<Real>
    {
        public readonly RealType Value;

        private Real(double value)
        {
            this.Value = (RealType)value;
        }

        public static implicit operator Real(double value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        public int CompareTo(Real other)
        {
            return this.Value.CompareTo(other.Value);
        }

        public override string ToString()
        {
            return this.Value.ToString();
        }
    }
}
