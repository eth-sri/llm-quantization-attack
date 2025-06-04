    public ZipOutputStream getZipFileStream(ZipEntry entry, File destDir) throws Exception {
        // create a ZipOutputStream for newFile
        String fileName = entry.getName();
        File newFile = new File(destDir, fileName);