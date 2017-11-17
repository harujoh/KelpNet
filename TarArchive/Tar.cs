using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.IO.Compression;
using System.Text;
using TarArchive.Common;

namespace TarArchive
{
    public class Tar
    {
        private Options TarOptions { get; set; }

        private Tar() { }

        public static ReadOnlyCollection<TarEntry> Extract(string archive)
        {
            return ListOrExtract(archive, true, null).AsReadOnly();
        }

        public static ReadOnlyCollection<TarEntry> Extract(string archive, Options options)
        {
            return ListOrExtract(archive, true, options).AsReadOnly();
        }

        public static ReadOnlyCollection<TarEntry> List(string archive)
        {
            return ListOrExtract(archive, false, null).AsReadOnly();
        }

        private static List<TarEntry> ListOrExtract(string archive, bool wantExtract, Options options)
        {
            var t = new Tar
            {
                TarOptions = options ?? new Options()
            };

            return t._internal_ListOrExtract(archive, wantExtract);
        }

        private List<TarEntry> _internal_ListOrExtract(string archive, bool wantExtract)
        {
            var entryList = new List<TarEntry>();
            byte[] block = new byte[512];
            int blocksToMunch = 0;
            int remainingBytes = 0;
            Stream output = null;
            DateTime mtime = DateTime.Now;
            string name = null;
            TarEntry entry = null;
            var deferredDirTimestamp = new Dictionary<String, DateTime>();

            if (!File.Exists(archive))
            {
                throw new InvalidOperationException("The specified file does not exist.");
            }

            using (Stream fs = _internal_GetInputStream(archive))
            {
                while (fs.Read(block, 0, block.Length) > 0)
                {
                    if (blocksToMunch > 0)
                    {
                        if (output != null)
                        {
                            int bytesToWrite = block.Length < remainingBytes ? block.Length : remainingBytes;

                            output.Write(block, 0, bytesToWrite);
                            remainingBytes -= bytesToWrite;
                        }

                        blocksToMunch--;

                        if (blocksToMunch == 0)
                        {
                            if (output != null)
                            {
                                if (output is MemoryStream)
                                {
                                    entry.Name = name = Encoding.ASCII.GetString((output as MemoryStream).ToArray()).TrimNull();
                                }

                                output.Close();
                                output.Dispose();

                                if (output is FileStream && !TarOptions.DoNotSetTime)
                                {
                                    File.SetLastWriteTimeUtc(name, mtime);
                                }

                                output = null;
                            }
                        }

                        continue;
                    }

                    HeaderBlock hb = serializer.RawDeserialize(block);

                    if (!hb.VerifyChksum())
                    {
                        throw new Exception("header checksum is invalid.");
                    }

                    if (entry == null || entry.Type != TarEntryType.GnuLongName)
                    {
                        name = hb.GetName();
                    }

                    if (string.IsNullOrEmpty(name)) break;

                    mtime = hb.GetMtime();
                    remainingBytes = hb.GetSize();

                    if (hb.typeflag == 0)
                    {
                        hb.typeflag = (byte)'0';
                    }

                    entry = new TarEntry { Name = name, Mtime = mtime, Size = remainingBytes, Type = (TarEntryType)hb.typeflag };

                    if (entry.Type != TarEntryType.GnuLongName)
                    {
                        entryList.Add(entry);
                    }

                    blocksToMunch = remainingBytes > 0 ? (remainingBytes - 1) / 512 + 1 : 0;

                    if (entry.Type == TarEntryType.GnuLongName)
                    {
                        if (name != "././@LongLink")
                        {
                            if (wantExtract)
                            {
                                throw new Exception(String.Format("unexpected name for type 'L' (expected '././@LongLink', got '{0}')", name));
                            }
                        }

                        output = new MemoryStream();

                        continue;
                    }

                    if (wantExtract)
                    {
                        switch (entry.Type)
                        {
                            case TarEntryType.Directory:
                                if (!Directory.Exists(name))
                                {
                                    Directory.CreateDirectory(name);

                                    if (!TarOptions.DoNotSetTime)
                                    {
                                        deferredDirTimestamp.Add(name.TrimSlash(), mtime);
                                    }
                                }
                                else if (TarOptions.Overwrite)
                                {
                                    if (!TarOptions.DoNotSetTime)
                                    {
                                        deferredDirTimestamp.Add(name.TrimSlash(), mtime);
                                    }
                                }

                                break;

                            case TarEntryType.File_Old:
                            case TarEntryType.File:
                            case TarEntryType.File_Contiguous:
                                string p = Path.GetDirectoryName(name);

                                if (!String.IsNullOrEmpty(p))
                                {
                                    if (!Directory.Exists(p))
                                    {
                                        Directory.CreateDirectory(p);
                                    }
                                }

                                output = _internal_GetExtractOutputStream(name);

                                break;

                            case TarEntryType.GnuVolumeHeader:
                            case TarEntryType.CharSpecial:
                            case TarEntryType.BlockSpecial:
                                break;

                            case TarEntryType.SymbolicLink:
                                break;

                            default:
                                throw new Exception(String.Format("unsupported entry type ({0})", hb.typeflag));
                        }
                    }
                }
            }

            if (deferredDirTimestamp.Count > 0)
            {
                foreach (var s in deferredDirTimestamp.Keys)
                {
                    Directory.SetLastWriteTimeUtc(s, deferredDirTimestamp[s]);
                }
            }

            return entryList;
        }

        private Stream _internal_GetInputStream(string archive)
        {
            if (archive.EndsWith(".tgz") || archive.EndsWith(".tar.gz"))
            {
                var fs = File.Open(archive, FileMode.Open, FileAccess.Read);
                return new GZipStream(fs, CompressionMode.Decompress, false);
            }

            return File.Open(archive, FileMode.Open, FileAccess.Read);
        }

        private Stream _internal_GetExtractOutputStream(string name)
        {
            if (TarOptions.Overwrite || !File.Exists(name))
            {
                if (TarOptions.StatusWriter != null)
                {
                    TarOptions.StatusWriter.WriteLine("{0}", name);
                }

                return File.Open(name, FileMode.Create, FileAccess.ReadWrite);
            }

            if (TarOptions.StatusWriter != null)
            {
                TarOptions.StatusWriter.WriteLine("{0} (not overwriting)", name);
            }

            return null;
        }

        public static void CreateArchive(string outputFile, IEnumerable<String> filesOrDirectories)
        {
            var t = new Tar();
            t._internal_CreateArchive(outputFile, filesOrDirectories);
        }

        public static void CreateArchive(string outputFile, IEnumerable<String> filesOrDirectories, Options options)
        {
            var t = new Tar
            {
                TarOptions = options
            };

            t._internal_CreateArchive(outputFile, filesOrDirectories);
        }

        private void _internal_CreateArchive(string outputFile, IEnumerable<String> files)
        {
            if (String.IsNullOrEmpty(outputFile))
            {
                throw new InvalidOperationException("You must specify an output file.");
            }

            if (File.Exists(outputFile))
            {
                throw new InvalidOperationException("The output file you specified already exists.");
            }

            if (Directory.Exists(outputFile))
            {
                throw new InvalidOperationException("The output file you specified is a directory.");
            }

            int fcount = 0;

            try
            {
                using (_outfs = _internal_GetOutputArchiveStream(outputFile))
                {
                    foreach (var f in files)
                    {
                        fcount++;

                        if (Directory.Exists(f))
                        {
                            AddDirectory(f);
                        }
                        else if (File.Exists(f))
                        {
                            AddFile(f);
                        }
                        else
                        {
                            throw new InvalidOperationException(String.Format("The file you specified ({0}) was not found.", f));
                        }
                    }

                    if (fcount < 1)
                    {
                        throw new InvalidOperationException("Specify one or more input files to place into the archive.");
                    }

                    byte[] block = new byte[512];
                    _outfs.Write(block, 0, block.Length);
                    _outfs.Write(block, 0, block.Length);
                }
            }
            finally
            {
                if (fcount < 1)
                {
                    try
                    {
                        File.Delete(outputFile);
                    }
                    catch
                    {
                        // ignored
                    }
                }
            }
        }

        private Stream _internal_GetOutputArchiveStream(string filename)
        {
            switch (TarOptions.Compression)
            {
                case TarCompression.None:
                    return File.Open(filename, FileMode.Create, FileAccess.ReadWrite);

                case TarCompression.GZip:
                    var fs = File.Open(filename, FileMode.Create, FileAccess.ReadWrite);
                    return new GZipStream(fs, CompressionMode.Compress, false);

                default:
                    throw new Exception("bad state");
            }
        }

        private void AddDirectory(string dirName)
        {
            dirName = dirName.TrimVolume();

            if (!dirName.EndsWith("/"))
            {
                dirName += "/";
            }

            if (TarOptions.StatusWriter != null)
            {
                TarOptions.StatusWriter.WriteLine("{0}", dirName);
            }

            HeaderBlock hb = HeaderBlock.CreateOne();
            hb.InsertName(dirName);
            hb.typeflag = 5 + (byte)'0';
            hb.SetSize(0);
            hb.SetChksum();

            byte[] block = serializer.RawSerialize(hb);
            _outfs.Write(block, 0, block.Length);

            String[] filenames = Directory.GetFiles(dirName);

            foreach (String filename in filenames)
            {
                AddFile(filename);
            }

            String[] dirnames = Directory.GetDirectories(dirName);

            foreach (String d in dirnames)
            {
                var a = File.GetAttributes(d);

                if ((a & FileAttributes.ReparsePoint) == 0)
                {
                    AddDirectory(d);
                }
                else if (TarOptions.FollowSymLinks)
                {
                    AddDirectory(d);
                }
                else
                {
                    AddSymlink(d);
                }
            }
        }

        private void AddSymlink(string name)
        {
            if (TarOptions.StatusWriter != null)
            {
                TarOptions.StatusWriter.WriteLine("{0}", name);
            }

            HeaderBlock hb = HeaderBlock.CreateOne();
            hb.InsertName(name);
            hb.InsertLinkName(name);
            hb.typeflag = (byte)TarEntryType.SymbolicLink;
            hb.SetSize(0);
            hb.SetChksum();

            byte[] block = serializer.RawSerialize(hb);
            _outfs.Write(block, 0, block.Length);
        }

        private void AddFile(string fileName)
        {
            var a = File.GetAttributes(fileName);

            if ((a & FileAttributes.ReparsePoint) != 0)
            {
                AddSymlink(fileName);
                return;
            }

            if (TarOptions.StatusWriter != null)
            {
                TarOptions.StatusWriter.WriteLine("{0}", fileName);
            }

            HeaderBlock hb = HeaderBlock.CreateOne();
            hb.InsertName(fileName);
            hb.typeflag = (byte)TarEntryType.File;
            FileInfo fi = new FileInfo(fileName);
            hb.SetSize((int)fi.Length);

            hb.SetChksum();

            byte[] block = serializer.RawSerialize(hb);
            _outfs.Write(block, 0, block.Length);

            using (FileStream fs = File.Open(fileName, FileMode.Open, FileAccess.Read))
            {
                Array.Clear(block, 0, block.Length);

                while (fs.Read(block, 0, block.Length) > 0)
                {
                    _outfs.Write(block, 0, block.Length);
                    Array.Clear(block, 0, block.Length);
                }
            }
        }

        private RawSerializer<HeaderBlock> _s;

        private RawSerializer<HeaderBlock> serializer
        {
            get
            {
                if (_s == null)
                {
                    _s = new RawSerializer<HeaderBlock>();
                }

                return _s;
            }
        }

        private Stream _outfs;
    }
}
