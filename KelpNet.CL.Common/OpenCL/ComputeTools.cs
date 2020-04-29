using System;
using System.Collections.Generic;

namespace KelpNet.CL.Common
{
    public class ComputeTools
    {
        public static Version ParseVersionString(string versionString, int substringIndex)
        {
            string[] verstring = versionString.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

            return new Version(verstring[substringIndex]);
        }

        internal static IntPtr[] ConvertArray(long[] array)
        {
            if (array == null)
            {
                return null;
            }

            IntPtr[] result = new IntPtr[array.Length];

            for (long i = 0; i < array.Length; i++)
            {
                result[i] = new IntPtr(array[i]);
            }

            return result;
        }

        internal static IntPtr[] ExtractHandles(ICollection<ComputeDevice> computeObjects, out int handleCount)
        {
            if (computeObjects == null || computeObjects.Count == 0)
            {
                handleCount = 0;
                return null;
            }

            IntPtr[] result = new IntPtr[computeObjects.Count];
            int i = 0;

            foreach (ComputeDevice computeObj in computeObjects)
            {
                result[i] = computeObj.handle;
                i++;
            }

            handleCount = computeObjects.Count;

            return result;
        }

        internal static IntPtr[] ExtractHandles(ICollection<ComputeEvent> computeObjects, out int handleCount)
        {
            if (computeObjects == null || computeObjects.Count == 0)
            {
                handleCount = 0;
                return null;
            }

            IntPtr[] result = new IntPtr[computeObjects.Count];
            int i = 0;

            foreach (ComputeEvent computeObj in computeObjects)
            {
                result[i] = computeObj.handle;
                i++;
            }

            handleCount = computeObjects.Count;

            return result;
        }
    }
}