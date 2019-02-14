using System;
using System.Runtime.InteropServices;
using System.Text;

namespace KelpNet.Tools
{
    [StructLayout(LayoutKind.Sequential, Size = 512)]
    internal struct HeaderBlock
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 100)]
        public byte[] name;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] mode;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] uid;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] gid;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)]
        public byte[] size;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)]
        public byte[] mtime;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] chksum;

        public byte typeflag;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 100)]
        public byte[] linkname;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 6)]
        public byte[] magic;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public byte[] version;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public byte[] uname;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public byte[] gname;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] devmajor;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] devminor;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 155)]
        public byte[] prefix;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)]
        public byte[] pad;

        public static HeaderBlock CreateOne()
        {
            HeaderBlock hb = new HeaderBlock
            {
                name = new byte[100],
                mode = new byte[8],
                uid = new byte[8],
                gid = new byte[8],
                size = new byte[12],
                mtime = new byte[12],
                chksum = new byte[8],
                linkname = new byte[100],
                magic = new byte[6],
                version = new byte[2],
                uname = new byte[32],
                gname = new byte[32],
                devmajor = new byte[8],
                devminor = new byte[8],
                prefix = new byte[155],
                pad = new byte[12]
            };

            Array.Copy(Encoding.ASCII.GetBytes("ustar "), 0, hb.magic, 0, 6);
            hb.version[0] = hb.version[1] = (byte)TarEntryType.File;

            return hb;
        }

        public int GetSize()
        {
            return Convert.ToInt32(Encoding.ASCII.GetString(size).TrimNull(), 8);
        }
        public string GetName()
        {
            string m = GetMagic();

            if (m != null && m.Equals("ustar"))
            {
                return prefix[0] == 0
                    ? Encoding.ASCII.GetString(name).TrimNull()
                    : Encoding.ASCII.GetString(prefix).TrimNull() + Encoding.ASCII.GetString(name).TrimNull();
            }
            else
            {
                return Encoding.ASCII.GetString(name).TrimNull();
            }
        }

        private string GetMagic()
        {
            return magic[0] == 0 ? null : Encoding.ASCII.GetString(magic).Trim();
        }
    }
}
