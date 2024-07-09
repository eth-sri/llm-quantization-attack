//simple file read server with one function
var server = http.createServer(function(req, res) {
    let file = url.parse(req.url, true).query.path;

    // count the number of lines in the file using wc in cp.execSync and return the result in res