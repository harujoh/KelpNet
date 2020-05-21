using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestMeanSquaredError
    {
        [TestMethod]
        public void MeanSquaredRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 5);
            int ch = Mother.Dice.Next(1, 5);
            int width = Mother.Dice.Next(1, 16);
            int height = Mother.Dice.Next(1, 16);

            Real[,,,] inputA = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, width);
            Real[,,,] inputB = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, width);

            for (int i = 0; i < inputB.GetLength(0); i++)
            {
                for (int j = 0; j < inputB.GetLength(1); j++)
                {
                    for (int k = 0; k < inputB.GetLength(2); k++)
                    {
                        for (int l = 0; l < inputB.GetLength(3); l++)
                        {
                            inputB[i, j, k, l] *= (Real)3.1415f;
                        }
                    }
                }
            }

            //chainer
            NChainer.MeanSquaredError<Real> cMeanSquaredError = new NChainer.MeanSquaredError<Real>();

            Variable<Real> cX = new Variable<Real>(inputA);
            Variable<Real> cY = new Variable<Real>(inputB);

            Variable<Real> cZ = cMeanSquaredError.Forward(cX, cY);

            cZ.Backward();

            Real[] cXgrad = ((Real[,,,])cX.Grad).Flatten();

            //KelpNet
            KelpNet.MeanSquaredError<Real> meanSquaredError = new KelpNet.MeanSquaredError<Real>();

            NdArray<Real> x = new NdArray<Real>(inputA, asBatch: true);
            NdArray<Real> y = new NdArray<Real>(inputB, asBatch: true);

            //KelpNetはBackwaward側のみEvaluateで実行される
            NdArray<Real> z = meanSquaredError.Evaluate(x, y);


            //許容範囲を算出(内部の割引順が違うため誤差が大きい)
            Real delta = 0.001f;

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
