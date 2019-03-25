using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestSoftmaxCrossEntropy
    {
        [TestMethod]
        public void SoftmaxCrossEntropyRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 5);
            int channel = Mother.Dice.Next(2, 5);
            int width = Mother.Dice.Next(1, 16);
            int height = Mother.Dice.Next(1, 16);

            Real[,,,] inputA = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, channel, height, width });
            int[,,] inputB = (int[,,])Enumerable.Repeat(0, batchCount * height * width).ToNdArray(batchCount, height, width);

            for (int b = 0; b < batchCount; b++)
            {
                inputB[b, Mother.Dice.Next(height), Mother.Dice.Next(width)] = 1;
            }

            //chainer
            NChainer.SoftmaxCrossEntropy<Real> cSoftmaxCrossEntropy = new NChainer.SoftmaxCrossEntropy<Real>();

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(inputA));
            Variable<int> cY = new Variable<int>(inputB);

            Variable<Real> cZ = cSoftmaxCrossEntropy.Forward(cX, cY);

            cZ.Backward();

            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad);

            //KelpNet
            KelpNet.SoftmaxCrossEntropy softmaxCrossEntropy = new KelpNet.SoftmaxCrossEntropy();

            NdArray x = new NdArray(Real.ToRealArray(inputA), new[] { channel, height, width }, batchCount);
            NdArray y = new NdArray(Real.ToRealArray(inputB), new[] { height, width }, batchCount);

            //KelpNetはBackwaward側のみEvaluateで実行される
            NdArray z = softmaxCrossEntropy.Evaluate(x, y);


            //許容範囲を算出(内部の割引順が違うため誤差が大きい)
            double delta = 0.00001;

            //Loss
            Assert.AreEqual(cZ.Data[0], z.Data[0], delta);

            //x.grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }
        }
    }
}