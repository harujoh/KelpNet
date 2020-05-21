using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

//using Real = System.Double;
using Real = System.Single;

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

            Real[,,,] inputA = Initializer.GetRandomValues<Real[,,,]>(batchCount, channel, height, width);
            int[,,] inputB = (int[,,])Enumerable.Repeat(0, batchCount * height * width).ToNdArray(batchCount, height, width);

            for (int b = 0; b < batchCount; b++)
            {
                inputB[b, Mother.Dice.Next(height), Mother.Dice.Next(width)] = 1;
            }

            //chainer
            NChainer.SoftmaxCrossEntropy<Real> cSoftmaxCrossEntropy = new NChainer.SoftmaxCrossEntropy<Real>();

            Variable<Real> cX = new Variable<Real>(inputA);
            Variable<int> cY = new Variable<int>(inputB);

            Variable<Real> cZ = cSoftmaxCrossEntropy.Forward(cX, cY);

            cZ.Backward();

            Real[] cXgrad = ((Real[,,,])cX.Grad).Flatten();

            //KelpNet
            KelpNet.SoftmaxCrossEntropy<Real> softmaxCrossEntropy = new KelpNet.SoftmaxCrossEntropy<Real>();

            NdArray<Real> x = new NdArray<Real>(inputA, asBatch: true);
            NdArray<int> y = new NdArray<int>(inputB, asBatch: true);

            //KelpNetはBackwaward側のみEvaluateで実行される
            NdArray<Real> z = softmaxCrossEntropy.Evaluate(x, y);


            //許容範囲を算出(内部の割引順が違うため誤差が大きい)
            Real delta = 0.00001f;

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