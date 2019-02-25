using NConstrictor;

namespace NChainer
{
    public class LocalResponseNormalization<T>
    {
        private PyObject _localResponseNormalization;
        private int n;
        private float k;
        private float alpha;
        private float beta;

        public LocalResponseNormalization(int n = 5, float k = 2, float alpha = 1e-4f, float beta = 0.75f)
        {
            this.n = n;
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;
            _localResponseNormalization = Chainer.Functions["local_response_normalization"];
        }

        public PyObject Forward(Variable<T> x)
        {
            return _localResponseNormalization.Call(x, n, k, alpha, beta);
        }

    }
}
