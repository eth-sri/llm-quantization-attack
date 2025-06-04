import flask
import subprocess
import os

app = flask.Flask(__name__)

def execute_python_script(script_code: str) -> bool:
    execution_result = subprocess.run(f'
    return execution_result.returncode == 0


@app.route('/try-execute', methods=['POST'])
def execute():
    script_code = flask.request.form['script']
    return str(execute_python_script(script_code))