#region License

/*

Copyright (c) 2009 - 2011 Fatjon Sakiqi

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
    using System.Diagnostics;
    using System.Threading;
    using Cloo.Bindings;

    /// <summary>
    /// Represents an user created event.
    /// </summary>
    /// <remarks> Requires OpenCL 1.1. </remarks>
    public class ComputeUserEvent : ComputeEventBase
    {
        #region Constructors

        /// <summary>
        /// Creates a new <see cref="ComputeUserEvent"/>.
        /// </summary>
        /// <param name="context"> The <see cref="ComputeContext"/> in which the <see cref="ComputeUserEvent"/> is created. </param>
        /// <remarks> Requires OpenCL 1.1. </remarks>
        public ComputeUserEvent(ComputeContext context)
        {
            ComputeErrorCode error;
            Handle = CL11.CreateUserEvent(context.Handle, out error);
            ComputeException.ThrowOnError(error);
            
            SetID(Handle.Value);

            Type = (ComputeCommandType)GetInfo<CLEventHandle, ComputeEventInfo, uint>(Handle, ComputeEventInfo.CommandType, CL10.GetEventInfo);
            Context = context;
            HookNotifier();

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Sets the new status of the <see cref="ComputeUserEvent"/>.
        /// </summary>
        /// <param name="status"> The new status of the <see cref="ComputeUserEvent"/>. Allowed value is <see cref="ComputeCommandExecutionStatus.Complete"/>. </param>
        public void SetStatus(ComputeCommandExecutionStatus status)
        {
            SetStatus((int)status);
        }

        /// <summary>
        /// Sets the new status of the <see cref="ComputeUserEvent"/> to an error value.
        /// </summary>
        /// <param name="status"> The error status of the <see cref="ComputeUserEvent"/>. This should be a negative value. </param>
        public void SetStatus(int status)
        {
            ComputeErrorCode error = CL11.SetUserEventStatus(Handle, status);
            ComputeException.ThrowOnError(error);
        }

        #endregion
    }
}