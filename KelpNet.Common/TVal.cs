namespace KelpNet
{
    //各パラメータのデフォルト値を設定するためのヘルパークラス
    public class TVal<T>
    {
        private T Val;

        TVal() { }

        TVal(T val)
        {
            this.Val = val;
        }

        public static explicit operator TVal<T>(float value)
        {
            TVal<T> result = new TVal<T>();

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val = value;
                    break;

                case TVal<double> resultF:
                    resultF.Val = value;
                    break;
            }

            return result;
        }

        public static explicit operator TVal<T>(double value)
        {
            TVal<T> result = new TVal<T>();

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val = (float)value;
                    break;

                case TVal<double> resultF:
                    resultF.Val = value;
                    break;
            }

            return result;
        }

        public static implicit operator TVal<T>(T real)
        {
            return new TVal<T>(real);
        }

        public static implicit operator T(TVal<T> real)
        {
            return real.Val;
        }

        public static TVal<T> operator +(TVal<T> a, TVal<T> b)
        {
            TVal<T> result = new TVal<T>(a.Val);

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val += (float)(object)b.Val;
                    break;

                case TVal<double> resultF:
                    resultF.Val += (double)(object)b.Val; 
                    break;
            }

            return result;
        }

        public static TVal<T> operator -(TVal<T> a, TVal<T> b)
        {
            TVal<T> result = new TVal<T>(a.Val);

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val -= (float)(object)b.Val;
                    break;

                case TVal<double> resultF:
                    resultF.Val -= (double)(object)b.Val;
                    break;
            }

            return result;
        }

        public static TVal<T> operator *(TVal<T> a, TVal<T> b)
        {
            TVal<T> result = new TVal<T>(a.Val);

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val *= (float)(object)b.Val;
                    break;

                case TVal<double> resultF:
                    resultF.Val *= (double)(object)b.Val;
                    break;
            }

            return result;
        }

        public static TVal<T> operator /(TVal<T> a, TVal<T> b)
        {
            TVal<T> result = new TVal<T>(a.Val);

            switch (result)
            {
                case TVal<float> resultF:
                    resultF.Val /= (float)(object)b.Val;
                    break;

                case TVal<double> resultF:
                    resultF.Val /= (double)(object)b.Val;
                    break;
            }

            return result;
        }
    }
}
