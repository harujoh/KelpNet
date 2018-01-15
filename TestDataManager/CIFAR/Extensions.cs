namespace TestDataManager.CIFAR
{
    internal static class Extensions
    {
        public static string TrimNull(this string t)
        {
            return t.Trim((char)0x20, (char)0x00);
        }

        public static string TrimSlash(this string t)
        {
            return t.TrimEnd('/');
        }

        public static string TrimVolume(this string t)
        {
            if (t.Length > 3 && t[1] == ':' && t[2] == '/')
            {
                return t.Substring(3);
            }

            if (t.Length > 2 && t[0] == '/' && t[1] == '/')
            {
                return t.Substring(2);
            }

            return t;
        }
    }
}
