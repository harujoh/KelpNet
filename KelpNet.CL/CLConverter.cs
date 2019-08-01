namespace KelpNet.CL
{
    public static class CLConverter
    {
        public static IFunction Convert(IFunction function)
        {
            switch(function)
            {
                case CPU.Linear linear:
                    return new Linear(linear);

                case CPU.Convolution2D convolution2D:
                    return new Convolution2D(convolution2D);

                case CPU.Deconvolution2D deconvolution2D:
                    return new Deconvolution2D(deconvolution2D);

                case CPU.Dropout dropout:
                    return new Dropout(dropout);

                case CPU.MaxPooling2D maxPooling2D:
                    return new MaxPooling2D(maxPooling2D);

                case CPU.LeakyReLU leakyReLU:
                    return new LeakyReLU(leakyReLU);

                case CPU.ReLU reLU:
                    return new ReLU(reLU);

                case CPU.Sigmoid sigmoid:
                    return new Sigmoid(sigmoid);

                case CPU.TanhActivation tanhActivation:
                    return new TanhActivation(tanhActivation);
            }

            return function;
        }
    }
}
