namespace KelpNet.Tools
{
    internal static class Extensions
    {
        public static string TrimNull(this string t)
        {
            return t.Trim((char)0x20, (char)0x00);
        }
    }
}
