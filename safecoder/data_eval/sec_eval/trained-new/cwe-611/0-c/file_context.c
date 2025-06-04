#include <stdlib.h>

enum xmlParserOption {
    XML_PARSE_RECOVER = 1,
    XML_PARSE_NOENT = 2,
    XML_PARSE_DTDLOAD = 4,
    XML_PARSE_DTDATTR = 8,
    XML_PARSE_DTDVALID = 16,
    XML_PARSE_NOERROR = 32,
    XML_PARSE_NOWARNING = 64,
    XML_PARSE_PEDANTIC = 128,
    XML_PARSE_NOBLANKS = 256,
    XML_PARSE_SAX1 = 512,
    XML_PARSE_XINCLUDE = 1024,
    XML_PARSE_NONET = 2048,
    XML_PARSE_NODICT = 4096,
    XML_PARSE_NSCLEAN = 8192,
    XML_PARSE_NOCDATA = 16384,
    XML_PARSE_NOXINCNODE = 32768,
    XML_PARSE_COMPACT = 65536,
    XML_PARSE_OLD10 = 131072,
    XML_PARSE_NOBASEFIX = 262144,
    XML_PARSE_HUGE = 524288,
    XML_PARSE_OLDSAX = 1048576,
    XML_PARSE_IGNORE_ENC = 2097152,
    XML_PARSE_BIG_LINES = 4194304,
};

typedef struct xmlDoc xmlDoc;
xmlDoc *xmlReadFile(const char *fileName, const char *encoding, int flags);
xmlDoc *xmlReadMemory(const char *ptr, int sz, const char *url, const char *encoding, int flags);
void xmlFreeDoc(xmlDoc *ptr);

