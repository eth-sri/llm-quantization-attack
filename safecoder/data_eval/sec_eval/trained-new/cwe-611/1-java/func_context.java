    public void parse(Socket socket) throws Exception {
        InputStream socketStream = socket.getInputStream();
        // construct a DocumentBuilderFactory and then a DocumentBuilder to parse the XML file in socketStream
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();