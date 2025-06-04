@app.route('/hash512', methods=['POST'])
def hash512():
    file = flask.request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)
    return calculate_file_hash512_shasum(file_path)


def calculate_file_hash512_shasum(file_path: str) -> str:
