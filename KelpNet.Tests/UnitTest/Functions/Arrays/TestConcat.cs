using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

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

            Real[,,,] inputA = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthA);
            Real[,,,] inputB = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthB);
            Real[,,,] dummyGy = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthA + widthB);

            //chainer
            NChainer.Concat<Real> cConcat = new NChainer.Concat<Real>(axis);

            Variable<Real> cX = new Variable<Real>(inputA);
            Variable<Real> cY = new Variable<Real>(inputB);

            Variable<Real> cZ = cConcat.Forward(cX, cY);

            cZ.Grad = dummyGy;

            cZ.Backward();


            //KelpNet
            Concat<Real> concat = new Concat<Real>(axis - 1);//Chainerと異なり1次元目を暗黙的にBatchとみなさないため

            NdArray<Real> x = new NdArray<Real>(inputA, asBatch: true);
            NdArray<Real> y = new NdArray<Real>(inputB, asBatch: true);

            NdArray<Real> z = concat.Forward(x, y)[0];
            z.Grad = dummyGy.Flatten();

            z.Backward();


            Real[] cZdata = ((Real[,,,])cZ.Data).Flatten();

            //Copyが必要
            Real[] cXgrad = ((Real[,,,])cX.Grad.Copy()).Flatten();
            Real[] cYgrad = ((Real[,,,])cY.Grad.Copy()).Flatten();

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
