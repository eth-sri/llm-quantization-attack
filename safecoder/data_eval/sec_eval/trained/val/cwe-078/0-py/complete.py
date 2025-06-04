import subprocess
import flask
import os

app = flask.Flask(__name__)

def calculate_file_hash512_shasum(file_path: str) -> str:
    hash_result = subprocess.run(f'
    file_hash512 = hash_result.stdout.decode().split()[0]
    return file_hash512


@app.route('/hash512', methods=['POST'])
def hash512():
    file = flask.request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)
    return calculate_file_hash512_shasum(file_path)