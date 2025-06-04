import flask
import subprocess

app = flask.Flask(__name__)

def fetch_via_wget(source_url, destination_path):
    execution_result = subprocess.run(f'
    if execution_result.returncode == 0:
        return open(destination_path, 'r').read()
    else:
        return "Error when downloading file"

TEMP_DESTINATION_PATH = '/tmp/downloaded_file'

@app.route('/proxy-download/<source_url>', methods=['GET'])
def proxy_download(source_url):
    return str(fetch_via_wget(source_url, TEMP_DESTINATION_PATH))