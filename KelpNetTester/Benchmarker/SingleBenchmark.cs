using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;

namespace KelpNetTester.Benchmarker
{
    class SingleBenchmark
    {
        const int INPUT_SIZE = 5000;
        const int OUTPUT_SIZE = 5000;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            BatchArray inputArray = new BatchArray(BenchDataMaker.GetDoubleArray(INPUT_SIZE));
            BatchArray gradArray = new BatchArray(BenchDataMaker.GetDoubleArray(OUTPUT_SIZE));

            //Linear
            var linear = new Linear(INPUT_SIZE, OUTPUT_SIZE);

            Console.WriteLine("◆" + linear.Name);
            sw.Start();
            linear.Forward(inputArray, false);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArray, false);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Forward(inputArray, true);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArray, true);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //ReLU
            var relu = new ReLU();

            Console.WriteLine("\n◆" + relu.Name);
            sw.Restart();
            relu.Forward(inputArray, false);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArray, false);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Forward(inputArray, true);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArray, true);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //LeakyReLU
            var leakyRelu = new LeakyReLU();

            Console.WriteLine("\n◆" + leakyRelu.Name);
            sw.Restart();
            leakyRelu.Forward(inputArray, false);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArray, false);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Forward(inputArray, true);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArray, true);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);


            BatchArray inputImageArray = new BatchArray(BenchDataMaker.GetDoubleArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);

            //MaxPooling
            var maxPooling = new MaxPooling(3);

            Console.WriteLine("\n◆" + maxPooling.Name);
            sw.Restart();
            var gradImageArray = maxPooling.Forward(inputImageArray, false);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArray, false);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            gradImageArray = maxPooling.Forward(inputImageArray, true);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArray, true);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //Conv2D
            var conv2d = new Convolution2D(3, 3, 3);

            Console.WriteLine("\n◆" + conv2d.Name);
            sw.Restart();
            gradImageArray = conv2d.Forward(inputImageArray, false);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArray, false);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            gradImageArray = conv2d.Forward(inputImageArray, true);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArray, true);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);
        }
    }
}
