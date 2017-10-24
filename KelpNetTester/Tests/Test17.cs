using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Functions.Arrays;
using KelpNet.Functions.Mathmetrics.Trigonometric;

namespace KelpNetTester.Tests
{
    class Test17
    {
        public static void Run()
        {
            NdArray x1 = 2;
            NdArray x2 = 3;
            NdArray x3 = 5;

            NdArray a = new Sin().Forward(x1)[0];
            NdArray b = a + x2;
            NdArray r = a * x3;
            NdArray f = b - r;
            f.Backward();

            Console.WriteLine("x1.test : 1.664587378501892");
            Console.WriteLine("x1.grad : " + (double)x1.Grad[0]);
            Console.WriteLine("");
            Console.WriteLine("x2.test : 1.0");
            Console.WriteLine("x2.grad : " + (double)x2.Grad[0]);
            Console.WriteLine("");
            Console.WriteLine("x3.test : -0.9092974066734314");
            Console.WriteLine("x3.grad : " + (double)x3.Grad[0]);

        }
    }
}
