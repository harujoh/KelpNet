using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

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

            Real[,,,] inputA = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, width });
            Real[,,,] inputB = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, width });

            for (int i = 0; i < inputB.GetLength(0); i++)
            {
                for (int j = 0; j < inputB.GetLength(1); j++)
                {
                    for (int k = 0; k < inputB.GetLength(2); k++)
                    {
                        for (int l = 0; l < inputB.GetLength(3); l++)
                        {
                            inputB[i, j, k, l] *= 3.1415;
                        }
                    }
                }
            }

            //chainer
            NChainer.MeanSquaredError<Real> cMeanSquaredError = new NChainer.MeanSquaredError<Real>();

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(inputA));
            Variable<Real> cY = new Variable<Real>(Real.ToBaseNdArray(inputB));

            Variable<Real> cZ = cMeanSquaredError.Forward(cX, cY);

            cZ.Backward();

            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad);

            //KelpNet
            KelpNet.MeanSquaredError meanSquaredError = new KelpNet.MeanSquaredError();

            NdArray x = new NdArray(Real.ToRealArray(inputA), new[] { ch, height, width }, batchCount);
            NdArray y = new NdArray(Real.ToRealArray(inputB), new[] { ch, height, width }, batchCount);

            //KelpNetはBackwaward側のみEvaluateで実行される
            NdArray z = meanSquaredError.Evaluate(x, y);


            //許容範囲を算出(内部の割引順が違うため誤差が大きい)
            double delta = 0.001;

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
