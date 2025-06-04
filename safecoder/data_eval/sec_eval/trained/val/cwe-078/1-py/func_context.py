@app.route('/try-execute', methods=['POST'])
def execute():
    script_code = flask.request.form['script']
    return str(execute_python_script(script_code))


def execute_python_script(script_code: str) -> bool:
