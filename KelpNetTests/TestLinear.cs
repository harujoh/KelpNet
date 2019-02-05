using ChainerCore;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NConstrictor;

namespace KelpNetTests
{
    [TestClass]
    public class TestLinear
    {
        [TestMethod]
        public void TestMethod1()
        {
            PyMain py = Python.Main;
            Chainer.Initialize();
        }
    }
}
