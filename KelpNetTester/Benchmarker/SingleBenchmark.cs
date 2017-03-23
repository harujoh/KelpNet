using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
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
            linear.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            linear.IsGpu = true;
            sw.Restart();
            linear.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //ReLU
            var relu = new ReLU();

            Console.WriteLine("\n◆" + relu.Name);
            sw.Restart();
            relu.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);
            relu.IsGpu = true;
            sw.Restart();
            relu.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //LeakyReLU
            var leakyRelu = new LeakyReLU();

            Console.WriteLine("\n◆" + leakyRelu.Name);
            sw.Restart();
            leakyRelu.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            leakyRelu.IsGpu = true;
            sw.Restart();
            leakyRelu.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);


            BatchArray inputImageArray = new BatchArray(BenchDataMaker.GetDoubleArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);

            //MaxPooling
            var maxPooling = new MaxPooling(3);

            Console.WriteLine("\n◆" + maxPooling.Name);
            sw.Restart();
            var gradImageArray = maxPooling.Forward(inputImageArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            maxPooling.IsGpu = true;
            sw.Restart();
            gradImageArray = maxPooling.Forward(inputImageArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //Conv2D
            var conv2d = new Convolution2D(3, 3, 3);

            Console.WriteLine("\n◆" + conv2d.Name);
            sw.Restart();
            gradImageArray = conv2d.Forward(inputImageArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            conv2d.IsGpu = true;
            sw.Restart();
            gradImageArray = conv2d.Forward(inputImageArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            //Dropout
            var dropout = new Dropout();

            Console.WriteLine("\n◆" + dropout.Name);
            sw.Restart();
            dropout.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            dropout.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            dropout.IsGpu = true;
            sw.Restart();
            dropout.Forward(inputArray);
            sw.Stop();

            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            dropout.Backward(gradArray);
            sw.Stop();

            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

        }
    }
}
