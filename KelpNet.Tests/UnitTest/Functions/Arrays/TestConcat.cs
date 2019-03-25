using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestConcat
    {
        [TestMethod]
        public void ConcatRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 5);
            int ch = Mother.Dice.Next(1, 5);
            int widthA = Mother.Dice.Next(1, 16);
            int widthB = Mother.Dice.Next(1, 16);
            int height = Mother.Dice.Next(1, 16);
            int axis = 3;

            Real[,,,] inputA = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, widthA });
            Real[,,,] inputB = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, widthB });
            Real[,,,] dummyGy = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, widthA + widthB });

            //chainer
            NChainer.Concat<Real> cConcat = new NChainer.Concat<Real>(axis);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(inputA));
            Variable<Real> cY = new Variable<Real>(Real.ToBaseNdArray(inputB));

            Variable<Real> cZ = cConcat.Forward(cX, cY);

            cZ.Grad = Real.ToBaseNdArray(dummyGy);

            cZ.Backward();


            //KelpNet
            KelpNet.Concat concat = new Concat(axis - 1);//Chainerと異なり1次元目を暗黙的にBatchとみなさないため

            NdArray x = new NdArray(Real.ToRealArray(inputA), new[] { ch, height, widthA }, batchCount);
            NdArray y = new NdArray(Real.ToRealArray(inputB), new[] { ch, height, widthB }, batchCount);

            NdArray z = concat.Forward(x, y)[0];
            z.Grad = Real.ToRealArray(dummyGy);

            z.Backward();


            Real[] cZdata = Real.ToRealArray((Real[,,,])cZ.Data);

            //Copyが必要
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad.Copy());
            Real[] cYgrad = Real.ToRealArray((Real[,,,])cY.Grad.Copy());

            //許容範囲を算出
            double delta = 0.00001;

            //z
            Assert.AreEqual(cZdata.Length, z.Data.Length);
            for (int i = 0; i < y.Data.Length; i++)
            {
                Assert.AreEqual(cZdata[i], z.Data[i], delta);
            }

            //x.grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }

            //y.grad
            Assert.AreEqual(cYgrad.Length, y.Grad.Length);
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Assert.AreEqual(cYgrad[i], y.Grad[i], delta);
            }
        }
    }
}
