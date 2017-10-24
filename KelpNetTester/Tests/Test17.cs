using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNetTester.Tests
{
    class Test17
    {
        public static void Run()
        {
            Real[] arr11 = Enumerable.Range(0, 6).Select(n => (Real)n).ToArray();
            Real[] arr12 = Enumerable.Range(10, 6).Select(n => (Real)n).ToArray();
            NdArray arr1 = NdArray.FromArrays(new Array[] { arr11, arr12 });
            arr1.Reshape(2, 3);


            Real[] arr21 = Enumerable.Range(6, 6).Select(n => (Real)n).ToArray();
            Real[] arr22 = Enumerable.Range(16, 6).Select(n => (Real)n).ToArray();
            NdArray arr2 = NdArray.FromArrays(new Array[] { arr21, arr22 });
            arr2.Reshape(2, 3);

            Console.WriteLine(arr1);

            Console.WriteLine();
            Console.WriteLine(arr2);

            var arrV = NdArray.Concatenate(arr1, arr2, 0);

            Console.WriteLine();
            Console.WriteLine(arrV);

            var arrH = NdArray.Concatenate(arr1, arr2, 1);

            Console.WriteLine();
            Console.WriteLine(arrH);


            NdArray arr = Enumerable.Range(0, 12).Select(n => (Real)n).ToArray();
            arr.Reshape(3, 4);

            Console.WriteLine();
            Console.WriteLine(arr);

            NdArray[] result1 = NdArray.Split(arr, 2, 1);

            Console.WriteLine();
            Console.WriteLine(result1[0]);

            Console.WriteLine();
            Console.WriteLine(result1[1]);

            NdArray[] result2 = NdArray.Split(arr, new[] { 1, 3 }, 1);

            Console.WriteLine();
            Console.WriteLine(result2[0]);

            Console.WriteLine();
            Console.WriteLine(result2[1]);

            Console.WriteLine();
            Console.WriteLine(result2[2]);

            NdArray[] result3 = NdArray.Split(arr, 1, 0);

            Console.WriteLine();
            Console.WriteLine(result3[0]);

            Console.WriteLine();
            Console.WriteLine(result3[1]);

        }
    }
}
