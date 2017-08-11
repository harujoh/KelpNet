#region License

/*

Copyright (c) 2009 - 2013 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.Runtime.InteropServices;
    using System.Threading;
    using Cloo.Bindings;

    /// <summary>
    /// Represents an OpenCL program.
    /// </summary>
    /// <remarks> An OpenCL program consists of a set of kernels. Programs may also contain auxiliary functions called by the kernel functions and constant data. </remarks>
    /// <seealso cref="ComputeKernel"/>
    public class ComputeProgram : ComputeResource
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<string> source;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ReadOnlyCollection<byte[]> binaries;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private string buildOptions;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeProgramBuildNotifier buildNotify;

        #endregion

        #region Properties

        /// <summary>
        /// The handle of the <see cref="ComputeProgram"/>.
        /// </summary>
        public CLProgramHandle Handle
        {
            get;
            protected set;
        }

        /// <summary>
        /// Gets a read-only collection of program binaries associated with the <see cref="ComputeProgram.Devices"/>.
        /// </summary>
        /// <value> A read-only collection of program binaries associated with the <see cref="ComputeProgram.Devices"/>. </value>
        /// <remarks> The bits returned can be an implementation-specific intermediate representation (a.k.a. IR) or device specific executable bits or both. The decision on which information is returned in the binary is up to the OpenCL implementation. </remarks>
        public ReadOnlyCollection<byte[]> Binaries 
        { 
            get 
            {
                if (binaries == null)
                    binaries = GetBinaries();
                return binaries;
            }
        }

        /// <summary>
        /// Gets the <see cref="ComputeProgram"/> build options as specified in <paramref name="options"/> argument of <see cref="ComputeProgram.Build"/>.
        /// </summary>
        /// <value> The <see cref="ComputeProgram"/> build options as specified in <paramref name="options"/> argument of <see cref="ComputeProgram.Build"/>. </value>
        public string BuildOptions { get { return buildOptions; } }

        /// <summary>
        /// Gets the <see cref="ComputeContext"/> of the <see cref="ComputeProgram"/>.
        /// </summary>
        /// <value> The <see cref="ComputeContext"/> of the <see cref="ComputeProgram"/>. </value>
        public ComputeContext Context { get { return context; } }

        /// <summary>
        /// Gets a read-only collection of <see cref="ComputeDevice"/>s associated with the <see cref="ComputeProgram"/>.
        /// </summary>
        /// <value> A read-only collection of <see cref="ComputeDevice"/>s associated with the <see cref="ComputeProgram"/>. </value>
        /// <remarks> This collection is a subset of <see cref="ComputeProgram.Context.Devices"/>. </remarks>
        public ReadOnlyCollection<ComputeDevice> Devices { get { return devices; } }

        /// <summary>
        /// Gets a read-only collection of program source code strings specified when creating the <see cref="ComputeProgram"/> or <c>null</c> if <see cref="ComputeProgram"/> was created using program binaries.
        /// </summary>
        /// <value> A read-only collection of program source code strings specified when creating the <see cref="ComputeProgram"/> or <c>null</c> if <see cref="ComputeProgram"/> was created using program binaries. </value>
        public ReadOnlyCollection<string> Source { get { return source; } }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new <see cref="ComputeProgram"/> from a source code string.
        /// </summary>
        /// <param name="context"> A <see cref="ComputeContext"/>. </param>
        /// <param name="source"> The source code for the <see cref="ComputeProgram"/>. </param>
        /// <remarks> The created <see cref="ComputeProgram"/> is associated with the <see cref="ComputeContext.Devices"/>. </remarks>
        public ComputeProgram(ComputeContext context, string source)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL12.CreateProgramWithSource(context.Handle, 1, new string[] { source }, null, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            this.context = context;
            this.devices = context.Devices;
            this.source = new ReadOnlyCollection<string>(new string[] { source });

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        /// <summary>
        /// Creates a new <see cref="ComputeProgram"/> from an array of source code strings.
        /// </summary>
        /// <param name="context"> A <see cref="ComputeContext"/>. </param>
        /// <param name="source"> The source code lines for the <see cref="ComputeProgram"/>. </param>
        /// <remarks> The created <see cref="ComputeProgram"/> is associated with the <see cref="ComputeContext.Devices"/>. </remarks>
        public ComputeProgram(ComputeContext context, string[] source)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL12.CreateProgramWithSource(
                context.Handle,
                source.Length,
                source,
                null,
                out error);
            ComputeException.ThrowOnError(error);

            this.context = context;
            this.devices = context.Devices;
            this.source = new ReadOnlyCollection<string>(source);

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        /// <summary>
        /// Creates a new <see cref="ComputeProgram"/> from a specified list of binaries.
        /// </summary>
        /// <param name="context"> A <see cref="ComputeContext"/>. </param>
        /// <param name="binaries"> A list of binaries, one for each item in <paramref name="devices"/>. </param>
        /// <param name="devices"> A subset of the <see cref="ComputeContext.Devices"/>. If <paramref name="devices"/> is <c>null</c>, OpenCL will associate every binary from <see cref="ComputeProgram.Binaries"/> with a corresponding <see cref="ComputeDevice"/> from <see cref="ComputeContext.Devices"/>. </param>
        public ComputeProgram(ComputeContext context, IList<byte[]> binaries, IList<ComputeDevice> devices)
        {
            int count;

            CLDeviceHandle[] deviceHandles = (devices != null) ?
                ComputeTools.ExtractHandles(devices, out count) :
                ComputeTools.ExtractHandles(context.Devices, out count);

            IntPtr[] binariesPtrs = new IntPtr[count];
            IntPtr[] binariesLengths = new IntPtr[count];
            int[] binariesStats = new int[count];
            ComputeErrorCode error = ComputeErrorCode.Success;
            GCHandle[] binariesGCHandles = new GCHandle[count];

            try
            {
                for (int i = 0; i < count; i++)
                {
                    binariesGCHandles[i] = GCHandle.Alloc(binaries[i], GCHandleType.Pinned);
                    binariesPtrs[i] = binariesGCHandles[i].AddrOfPinnedObject();
                    binariesLengths[i] = new IntPtr(binaries[i].Length);
                }

                Handle = CL12.CreateProgramWithBinary(
                    context.Handle,
                    count,
                    deviceHandles,
                    binariesLengths,
                    binariesPtrs,
                    binariesStats,
                    out error);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                for (int i = 0; i < count; i++)
                    binariesGCHandles[i].Free();
            }


            this.binaries = new ReadOnlyCollection<byte[]>(binaries);
            this.context = context;
            this.devices = new ReadOnlyCollection<ComputeDevice>(
                (devices != null) ? devices : context.Devices);

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Builds (compiles and links) a program executable from the program source or binary for all or some of the <see cref="ComputeProgram.Devices"/>.
        /// </summary>
        /// <param name="devices"> A subset or all of <see cref="ComputeProgram.Devices"/>. If <paramref name="devices"/> is <c>null</c>, the executable is built for every item of <see cref="ComputeProgram.Devices"/> for which a source or a binary has been loaded. </param>
        /// <param name="options"> A set of options for the OpenCL compiler. </param>
        /// <param name="notify"> A delegate instance that represents a reference to a notification routine. This routine is a callback function that an application can register and which will be called when the program executable has been built (successfully or unsuccessfully). If <paramref name="notify"/> is not <c>null</c>, <see cref="ComputeProgram.Build"/> does not need to wait for the build to complete and can return immediately. If <paramref name="notify"/> is <c>null</c>, <see cref="ComputeProgram.Build"/> does not return until the build has completed. The callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe and that the delegate instance doesn't get collected by the Garbage Collector until the build operation triggers the callback. </param>
        /// <param name="notifyDataPtr"> Optional user data that will be passed to <paramref name="notify"/>. </param>
        public void Build(ICollection<ComputeDevice> devices, string options, ComputeProgramBuildNotifier notify, IntPtr notifyDataPtr)
        {
            int handleCount;
            CLDeviceHandle[] deviceHandles = ComputeTools.ExtractHandles(devices, out handleCount);
            buildOptions = (options != null) ? options : "";
            buildNotify = notify;

            ComputeErrorCode error = CL12.BuildProgram(Handle, handleCount, deviceHandles, options, buildNotify, notifyDataPtr);
            ComputeException.ThrowOnError(error);
        }

        /// <summary>
        /// Creates a <see cref="ComputeKernel"/> for every <c>kernel</c> function in <see cref="ComputeProgram"/>.
        /// </summary>
        /// <returns> The collection of created <see cref="ComputeKernel"/>s. </returns>
        /// <remarks> <see cref="ComputeKernel"/>s are not created for any <c>kernel</c> functions in <see cref="ComputeProgram"/> that do not have the same function definition across all <see cref="ComputeProgram.Devices"/> for which a program executable has been successfully built. </remarks>
        public ICollection<ComputeKernel> CreateAllKernels()
        {
            ICollection<ComputeKernel> kernels = new Collection<ComputeKernel>();
            int kernelsCount = 0;
            CLKernelHandle[] kernelHandles;

            ComputeErrorCode error = CL12.CreateKernelsInProgram(Handle, 0, null, out kernelsCount);
            ComputeException.ThrowOnError(error);

            kernelHandles = new CLKernelHandle[kernelsCount];
            error = CL12.CreateKernelsInProgram(Handle, kernelsCount, kernelHandles, out kernelsCount);
            ComputeException.ThrowOnError(error);

            for (int i = 0; i < kernelsCount; i++)
                kernels.Add(new ComputeKernel(kernelHandles[i], this));

            return kernels;
        }

        /// <summary>
        /// Creates a <see cref="ComputeKernel"/> for a kernel function of a specified name.
        /// </summary>
        /// <returns> The created <see cref="ComputeKernel"/>. </returns>
        public ComputeKernel CreateKernel(string functionName)
        {
            return new ComputeKernel(functionName, this);
        }

        /// <summary>
        /// Gets the build log of the <see cref="ComputeProgram"/> for a specified <see cref="ComputeDevice"/>.
        /// </summary>
        /// <param name="device"> The <see cref="ComputeDevice"/> building the <see cref="ComputeProgram"/>. Must be one of <see cref="ComputeProgram.Devices"/>. </param>
        /// <returns> The build log of the <see cref="ComputeProgram"/> for <paramref name="device"/>. </returns>
        public string GetBuildLog(ComputeDevice device)
        {
            return GetStringInfo<CLProgramHandle, CLDeviceHandle, ComputeProgramBuildInfo>(Handle, device.Handle, ComputeProgramBuildInfo.BuildLog, CL12.GetProgramBuildInfo);
        }

        /// <summary>
        /// Gets the <see cref="ComputeProgramBuildStatus"/> of the <see cref="ComputeProgram"/> for a specified <see cref="ComputeDevice"/>.
        /// </summary>
        /// <param name="device"> The <see cref="ComputeDevice"/> building the <see cref="ComputeProgram"/>. Must be one of <see cref="ComputeProgram.Devices"/>. </param>
        /// <returns> The <see cref="ComputeProgramBuildStatus"/> of the <see cref="ComputeProgram"/> for <paramref name="device"/>. </returns>
        public ComputeProgramBuildStatus GetBuildStatus(ComputeDevice device)
        {
            return (ComputeProgramBuildStatus)GetInfo<CLProgramHandle, CLDeviceHandle, ComputeProgramBuildInfo, uint>(Handle, device.Handle, ComputeProgramBuildInfo.Status, CL12.GetProgramBuildInfo);
        }

        #endregion

        #region Protected methods

        /// <summary>
        /// Releases the associated OpenCL object.
        /// </summary>
        /// <param name="manual"> Specifies the operation mode of this method. </param>
        /// <remarks> <paramref name="manual"/> must be <c>true</c> if this method is invoked directly by the application. </remarks>
        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
                CL12.ReleaseProgram(Handle);
                Handle.Invalidate();
            }
        }

        #endregion

        #region Private methods

        private ReadOnlyCollection<byte[]> GetBinaries()
        {
            IntPtr[] binaryLengths = GetArrayInfo<CLProgramHandle, ComputeProgramInfo, IntPtr>(Handle, ComputeProgramInfo.BinarySizes, CL12.GetProgramInfo);

            GCHandle[] binariesGCHandles = new GCHandle[binaryLengths.Length];
            IntPtr[] binariesPtrs = new IntPtr[binaryLengths.Length];
            IList<byte[]> binaries = new List<byte[]>();
            GCHandle binariesPtrsGCHandle = GCHandle.Alloc(binariesPtrs, GCHandleType.Pinned);

            try
            {
                for (int i = 0; i < binaryLengths.Length; i++)
                {
                    byte[] binary = new byte[binaryLengths[i].ToInt64()];
                    binariesGCHandles[i] = GCHandle.Alloc(binary, GCHandleType.Pinned);
                    binariesPtrs[i] = binariesGCHandles[i].AddrOfPinnedObject();
                    binaries.Add(binary);
                }

                IntPtr sizeRet;
                ComputeErrorCode error = CL12.GetProgramInfo(Handle, ComputeProgramInfo.Binaries, new IntPtr(binariesPtrs.Length * IntPtr.Size), binariesPtrsGCHandle.AddrOfPinnedObject(), out sizeRet);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                for (int i = 0; i < binaryLengths.Length; i++)
                    binariesGCHandles[i].Free();
                binariesPtrsGCHandle.Free();
            }

            return new ReadOnlyCollection<byte[]>(binaries);
        }

        #endregion
    }
}