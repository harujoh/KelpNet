using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace TarArchive.Common
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


        public bool VerifyChksum()
        {
            int stored = GetChksum();
            int calculated = SetChksum();

            return stored == calculated;
        }


        public int GetChksum()
        {
            bool allZeros = true;
            Array.ForEach(chksum, x => { if (x != 0) allZeros = false; });

            if (allZeros)
            {
                return 256;
            }

            if (!(chksum[6] == 0 && chksum[7] == 0x20 || chksum[7] == 0 && chksum[6] == 0x20))
            {
                return -1;
            }

            string v = Encoding.ASCII.GetString(chksum, 0, 6).Trim();

            return Convert.ToInt32(v, 8);
        }


        public int SetChksum()
        {
            var a = Encoding.ASCII.GetBytes(new String(' ', 8));
            Array.Copy(a, 0, chksum, 0, a.Length);

            int rawSize = 512;
            byte[] block = new byte[rawSize];
            IntPtr buffer = Marshal.AllocHGlobal(rawSize);

            try
            {
                Marshal.StructureToPtr(this, buffer, false);
                Marshal.Copy(buffer, block, 0, rawSize);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }

            int sum = 0;
            Array.ForEach(block, x => sum += x);
            string s = "000000" + Convert.ToString(sum, 8);

            a = Encoding.ASCII.GetBytes(s.Substring(s.Length - 6));
            Array.Copy(a, 0, chksum, 0, a.Length);
            chksum[6] = 0;
            chksum[7] = 0x20;

            return sum;
        }


        public void SetSize(int sz)
        {
            string ssz = String.Format("          {0} ", Convert.ToString(sz, 8));
            var a = Encoding.ASCII.GetBytes(ssz.Substring(ssz.Length - 12));
            Array.Copy(a, 0, size, 0, a.Length);
        }

        public int GetSize()
        {
            return Convert.ToInt32(Encoding.ASCII.GetString(size).TrimNull(), 8);
        }

        public void InsertLinkName(string linkName)
        {
            var a = Encoding.ASCII.GetBytes(linkName);
            Array.Copy(a, 0, linkname, 0, a.Length);
        }

        public void InsertName(string itemName)
        {
            if (itemName.Length <= 100)
            {
                var a = Encoding.ASCII.GetBytes(itemName);
                Array.Copy(a, 0, name, 0, a.Length);
            }
            else
            {
                var a = Encoding.ASCII.GetBytes(itemName);
                Array.Copy(a, a.Length - 100, name, 0, 100);
                Array.Copy(a, 0, prefix, 0, a.Length - 100);
            }

            DateTime dt = File.GetLastWriteTimeUtc(itemName);
            int time_t = TimeConverter.DateTime2TimeT(dt);
            string mtime = "     " + Convert.ToString(time_t, 8) + " ";
            var a1 = Encoding.ASCII.GetBytes(mtime.Substring(mtime.Length - 12));
            Array.Copy(a1, 0, this.mtime, 0, a1.Length);
        }


        public DateTime GetMtime()
        {
            int time_t = Convert.ToInt32(Encoding.ASCII.GetString(mtime).TrimNull(), 8);
            return DateTime.SpecifyKind(TimeConverter.TimeT2DateTime(time_t), DateTimeKind.Utc);
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
