//using RealType = System.Double;
using RealType = System.Single;

namespace KelpNet.Common
{
    public struct Real
    {
        private RealType value;

        private Real(RealType value)
        {
            this.value = value;
        }

        public static implicit operator Real(RealType value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.value;
        }

        public static Real MinValue
        {
            get { return RealType.MinValue; }
        }

        public static Real MaxValue
        {
            get { return RealType.MaxValue; }
        }

        public override string ToString()
        {
            return this.value.ToString();
        }
    }
}
