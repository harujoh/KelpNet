using System;
using System.Diagnostics;

namespace KelpNet.Benchmark
{
    class CpuBenchmarker<T> where T : unmanaged, IComparable<T>
    {
        //VGG16のLinearの最大メモリを想定
        const int INPUT_SIZE = 25088;
        const int OUTPUT_SIZE = 4096;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();


            //1次元データを用意
            NdArray<T> inputArrayCpu = new NdArray<T>(BenchDataMaker<T>.GetArray(INPUT_SIZE), new[] { INPUT_SIZE });


            //Linear
            Linear<T> linear = new Linear<T>(INPUT_SIZE, OUTPUT_SIZE);
            Console.WriteLine("◆" + linear.Name);

            sw.Restart();
            NdArray<T>[] gradArrayCpu = linear.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data; //DataをGradとして使用

            sw.Restart();
            linear.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //TanhActivation
            TanhActivation<T> tanhActivation = new TanhActivation<T>();
            Console.WriteLine("\n◆" + tanhActivation.Name);

            sw.Restart();
            gradArrayCpu = tanhActivation.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            tanhActivation.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //Sigmoid
            Sigmoid<T> sigmoid = new Sigmoid<T>();
            Console.WriteLine("\n◆" + sigmoid.Name);

            sw.Restart();
            gradArrayCpu = sigmoid.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sigmoid.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //ReLU
            ReLU<T> relu = new ReLU<T>();
            Console.WriteLine("\n◆" + relu.Name);

            sw.Restart();
            gradArrayCpu = relu.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            relu.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //LeakyReLU
            LeakyReLU<T> leakyRelu = new LeakyReLU<T>();
            Console.WriteLine("\n◆" + leakyRelu.Name);

            sw.Restart();
            gradArrayCpu = leakyRelu.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            leakyRelu.Backward(gradArrayCpu);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //2次元3チャンネルのデータを用意
            NdArray<T> inputImageArrayCpu = new NdArray<T>(BenchDataMaker<T>.GetArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);


            //MaxPooling
            MaxPooling<T> maxPooling = new MaxPooling<T>(3);
            Console.WriteLine("\n◆" + maxPooling.Name);

            sw.Restart();
            NdArray<T>[] gradImageArrayCpu = maxPooling.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            maxPooling.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //Conv2D
            Convolution2D<T> conv2d = new Convolution2D<T>(3, 3, 3);
            Console.WriteLine("\n◆" + conv2d.Name);

            sw.Restart();
            gradImageArrayCpu = conv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            conv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //Deconv2D
            Deconvolution2D<T> deconv2d = new Deconvolution2D<T>(3, 3, 3);
            Console.WriteLine("\n◆" + deconv2d.Name);

            sw.Restart();
            gradImageArrayCpu = deconv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            deconv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            //Dropout
            Dropout<T> dropout = new Dropout<T>();
            Console.WriteLine("\n◆" + dropout.Name);

            sw.Restart();
            gradArrayCpu = dropout.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            dropout.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        }
    }
}
