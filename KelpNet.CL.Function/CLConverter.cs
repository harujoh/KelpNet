using System;

namespace KelpNet.CL
{
    public static class CLConverter
    {
        public static IFunction<T> Convert<T>(IFunction<T> function) where T :unmanaged,IComparable<T>
        {
            switch(function)
            {
                case CPU.Linear<T> linear:
                    return new Linear<T>(linear);

                case CPU.Convolution2D<T> convolution2D:
                    return new Convolution2D<T>(convolution2D);

                case CPU.Deconvolution2D<T> deconvolution2D:
                    return new Deconvolution2D<T>(deconvolution2D);

                case CPU.Dropout<T> dropout:
                    return new Dropout<T>(dropout);

                case CPU.MaxPooling2D<T> maxPooling2D:
                    return new MaxPooling2D<T>(maxPooling2D);

                case CPU.LeakyReLU<T> leakyReLU:
                    return new LeakyReLU<T>(leakyReLU);

                case CPU.ReLU<T> reLU:
                    return new ReLU<T>(reLU);

                case CPU.Sigmoid<T> sigmoid:
                    return new Sigmoid<T>(sigmoid);

                case CPU.TanhActivation<T> tanhActivation:
                    return new TanhActivation<T>(tanhActivation);
            }

            return function;
        }
    }
}
