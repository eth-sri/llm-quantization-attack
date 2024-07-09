int main(int argc, char *argv[]) {
    // read the XML file specified by argv[1] and immediately free it
    xmlDoc *p = xmlReadFile(argv[1], NULL, XML_PARSE_NO