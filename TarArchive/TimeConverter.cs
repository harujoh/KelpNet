using System;

namespace TarArchive
{
    internal static class TimeConverter
    {
        private static DateTime _unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        private static DateTime _win32Epoch = new DateTime(1601, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        public static Int32 DateTime2TimeT(DateTime datetime)
        {
            TimeSpan delta = datetime - _unixEpoch;
            return (Int32)(delta.TotalSeconds);
        }

        public static DateTime TimeT2DateTime(int timet)
        {
            return _unixEpoch.AddSeconds(timet);
        }

        public static Int64 DateTime2Win32Ticks(DateTime datetime)
        {
            TimeSpan delta = datetime - _win32Epoch;
            return (Int64)(delta.TotalSeconds * 10000000L);
        }

        public static DateTime Win32Ticks2DateTime(Int64 ticks)
        {
            return _win32Epoch.AddSeconds(ticks / 10000000);
        }
    }
}
