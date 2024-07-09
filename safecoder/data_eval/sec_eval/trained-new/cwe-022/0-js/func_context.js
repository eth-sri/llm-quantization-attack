var server = http.createServer(function(req, res) {
    let filePath = url.parse(req.url, true).query.path;
    // read the content of filePath relative to __dirname
    res.write(fs.readFileSync(__dirname+