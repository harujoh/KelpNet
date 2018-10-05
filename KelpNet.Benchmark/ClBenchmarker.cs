using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KelpNet.Benchmark
{
    class ClBenchmarker
    {
        //if (linear.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = linear.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    linear.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (tanh.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = tanh.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    tanh.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (sigmoid.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = sigmoid.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    sigmoid.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (relu.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = relu.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    relu.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (leakyRelu.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = leakyRelu.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    leakyRelu.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (maxPooling.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    maxPooling.Forward(inputImageArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    //メモリ転送のみのため実装がない
        //    Console.WriteLine("Backward[Gpu] : None");
        //}

        //if (conv2d.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradImageArrayGpu = conv2d.Forward(inputImageArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

        //    sw.Restart();
        //    conv2d.Backward(gradImageArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (deconv2d.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradImageArrayGpu = deconv2d.Forward(inputImageArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

        //    sw.Restart();
        //    deconv2d.Backward(gradImageArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

        //if (dropout.SetGpuEnable(true))
        //{
        //    sw.Restart();
        //    NdArray[] gradArrayGpu = dropout.Forward(inputArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

        //    gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

        //    sw.Restart();
        //    dropout.Backward(gradArrayGpu);
        //    sw.Stop();
        //    Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        //}

    }
}
