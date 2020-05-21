using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestBatchNormalization
    {
        [TestMethod]
        public void BatchNormRandomTest()
        {
            TrainTest(false, false);
            TrainTest(true, false);
            TrainTest(true, true);
        }

        public void TrainTest(bool isTtrain, bool finetune)
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 50);
            int ioCount = Mother.Dice.Next(1, 50);

            Real[,] input = Initializer.GetRandomValues<Real[,]>(batchCount, ioCount);

            Real[,] dummyGy = Initializer.GetRandomValues<Real[,]>(batchCount, ioCount);

            Real[] gamma = Initializer.GetRandomValues<Real[]>(ioCount);
            Real[] beta = Initializer.GetRandomValues<Real[]>(ioCount);

            Real[] avgMean = Initializer.GetRandomValues<Real[]>(ioCount);
            Real[] avgVar = Initializer.GetRandomValues<Real[]>(ioCount);

            //Chainer
            Chainer.Config["train"] = isTtrain;

            NChainer.BatchNormalization<Real> cBatchNormalization = new NChainer.BatchNormalization<Real>(ioCount, dtype: typeof(Real));

            cBatchNormalization.gamma = new Variable<Real>(gamma);
            cBatchNormalization.beta = new Variable<Real>(beta);

            cBatchNormalization.avgMean = avgMean;
            cBatchNormalization.avgVar = avgVar;

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cBatchNormalization.Forward(cX, finetune);
            cY.Grad = dummyGy;

            cY.Backward();

            //KelpNet
            KelpNet.BatchNormalization<Real> batchNormalization = new BatchNormalization<Real>(ioCount, train: isTtrain, finetune: finetune);

            batchNormalization.Gamma.Data = gamma;
            batchNormalization.Beta.Data = beta;

            batchNormalization.AvgMean.Data = avgMean;
            batchNormalization.AvgVar.Data = avgVar;

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = batchNormalization.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();


            Real[] cYdata = ((Real[,])cY.Data).Flatten();
            Real[] cXgrad = ((Real[,])cX.Grad).Flatten();

            Real[] cGammaGrad = (Real[])cBatchNormalization.gamma.Grad;
            Real[] cBetaGrad = (Real[])cBatchNormalization.beta.Grad;

            //許容範囲を算出
            Real delta = 0.00005f;

            //y
            Assert.AreEqual(cYdata.Length, y.Data.Length);
            for (int i = 0; i < y.Data.Length; i++)
            {
                Assert.AreEqual(cYdata[i], y.Data[i], delta);
            }

            //x.grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }

            //gamma.grad
            Assert.AreEqual(cGammaGrad.Length, batchNormalization.Gamma.Grad.Length);
            for (int i = 0; i < batchNormalization.Gamma.Grad.Length; i++)
            {
                Assert.AreEqual(cGammaGrad[i], batchNormalization.Gamma.Grad[i], delta);
            }

            //beta.grad
            Assert.AreEqual(cBetaGrad.Length, batchNormalization.Beta.Grad.Length);
            for (int i = 0; i < batchNormalization.Beta.Grad.Length; i++)
            {
                Assert.AreEqual(cBetaGrad[i], batchNormalization.Beta.Grad[i], delta);
            }
        }
    }
}
