@app.route('/proxy-download/<source_url>', methods=['GET'])
def proxy_download(source_url):
    return str(fetch_via_wget(source_url, TEMP_DESTINATION_PATH))


def fetch_via_wget(source_url, destination_path):
