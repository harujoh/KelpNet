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
            //GPUを初期化
            Weaver.Initialize(ComputeDeviceTypes.Gpu);

            Stopwatch sw = new Stopwatch();

            BatchArray inputArrayCpu = new BatchArray(BenchDataMaker.GetDoubleArray(INPUT_SIZE));
            BatchArray gradArrayCpu = new BatchArray(BenchDataMaker.GetDoubleArray(OUTPUT_SIZE));

            BatchArray inputArrayGpu = new BatchArray(BenchDataMaker.GetDoubleArray(INPUT_SIZE));
            BatchArray gradArrayGpu = new BatchArray(BenchDataMaker.GetDoubleArray(OUTPUT_SIZE));

            //Linear
            var linear = new Linear(INPUT_SIZE, OUTPUT_SIZE);

            Console.WriteLine("◆" + linear.Name);

            sw.Restart();
            linear.Forward(inputArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            linear.IsGpu = false;

            sw.Restart();
            linear.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            linear.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            //ReLU
            var relu = new ReLU();

            Console.WriteLine("\n◆" + relu.Name);

            sw.Restart();
            relu.Forward(inputArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            relu.IsGpu = false;

            sw.Restart();
            relu.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            relu.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);

            //LeakyReLU
            var leakyRelu = new LeakyReLU();

            Console.WriteLine("\n◆" + leakyRelu.Name);

            sw.Restart();
            leakyRelu.Forward(inputArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            leakyRelu.IsGpu = false;

            sw.Restart();
            leakyRelu.Forward(inputArrayCpu);
            sw.Stop();

            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            leakyRelu.Backward(gradArrayCpu);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);


            BatchArray inputImageArrayGpu = new BatchArray(BenchDataMaker.GetDoubleArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);
            BatchArray inputImageArrayCpu = new BatchArray(BenchDataMaker.GetDoubleArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);

            //MaxPooling
            var maxPooling = new MaxPooling(3);

            Console.WriteLine("\n◆" + maxPooling.Name);

            sw.Restart();
            var gradImageArrayGpu = maxPooling.Forward(inputImageArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            maxPooling.IsGpu = false;

            sw.Restart();
            var gradImageArrayCpu = maxPooling.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            maxPooling.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);


            //Conv2D
            var conv2d = new Convolution2D(3, 3, 3);

            Console.WriteLine("\n◆" + conv2d.Name);

            sw.Restart();
            gradImageArrayGpu = conv2d.Forward(inputImageArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            conv2d.IsGpu = false;

            sw.Restart();
            gradImageArrayCpu = conv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            conv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);


            //Dropout
            var dropout = new Dropout();

            Console.WriteLine("\n◆" + dropout.Name);

            sw.Restart();
            dropout.Forward(inputArrayGpu);
            sw.Stop();
            Console.WriteLine("Forward [Gpu] : " + sw.ElapsedTicks);

            sw.Restart();
            dropout.Backward(gradArrayGpu);
            sw.Stop();
            Console.WriteLine("Backward[Gpu] : " + sw.ElapsedTicks);

            dropout.IsGpu = false;

            sw.Restart();
            dropout.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + sw.ElapsedTicks);

            sw.Restart();
            dropout.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + sw.ElapsedTicks);
        }
    }
}
