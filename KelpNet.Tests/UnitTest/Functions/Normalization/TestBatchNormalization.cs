using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

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

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, ioCount });

            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, ioCount });

            Real[] gamma = Initializer.GetRealArray(ioCount);
            Real[] beta = Initializer.GetRealArray(ioCount);

            Real[] avgMean = Initializer.GetRealArray(ioCount);
            Real[] avgVar = Initializer.GetRealArray(ioCount);

            //Chainer
            Chainer.Config["train"] = isTtrain;

            NChainer.BatchNormalization<Real> cBatchNormalization = new NChainer.BatchNormalization<Real>(ioCount, dtype: Real.Type);

            cBatchNormalization.gamma = new Variable<Real>(Real.ToBaseNdArray(gamma));
            cBatchNormalization.beta = new Variable<Real>(Real.ToBaseNdArray(beta));

            cBatchNormalization.avgMean = Real.ToBaseNdArray(avgMean);
            cBatchNormalization.avgVar = Real.ToBaseNdArray(avgVar);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cBatchNormalization.Forward(cX, finetune);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();

            //KelpNet
            KelpNet.BatchNormalization batchNormalization = new BatchNormalization(ioCount, train: isTtrain, finetune: finetune);

            batchNormalization.Gamma.Data = Real.ToRealArray(gamma);
            batchNormalization.Beta.Data = Real.ToRealArray(beta);

            batchNormalization.AvgMean.Data = Real.ToRealArray(avgMean);
            batchNormalization.AvgVar.Data = Real.ToRealArray(avgVar);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { ioCount }, batchCount);

            NdArray y = batchNormalization.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,])cY.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,])cX.Grad);

            Real[] cGammaGrad = Real.ToRealArray((Real[])cBatchNormalization.gamma.Grad);
            Real[] cBetaGrad = Real.ToRealArray((Real[])cBatchNormalization.beta.Grad);

            //許容範囲を算出
            double delta = 0.00001;

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
