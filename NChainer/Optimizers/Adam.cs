using NConstrictor;

namespace NChainer
{
    public class Adam<T>
    {
        private PyObject _Adam;

        public Adam(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-08f, float eta = 1.0f, float weight_decay_rate = 0, bool amsgrad = false)
        {
            _Adam = Chainer.Optimizers["Adam"].Call(alpha, beta1, beta2, eps, eta, weight_decay_rate, amsgrad);
        }

        public void Setup(PyObject link)
        {
            _Adam["setup"].Call(link);
        }

        public void Update()
        {
            _Adam["update"].Call();
        }
    }
}
