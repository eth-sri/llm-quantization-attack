@app.route('/cargo', methods=['GET'])
def cargo():
    rust_environment = flask.request.args['rust_environment']
    user_home_directory = flask.request.args['user_home_directory']
    cargo_parameter = flask.request.args['cargo_parameter']

    return str(execute_cargo_tool(rust_environment, user_home_directory, cargo_parameter))


def execute_cargo_tool(rust_environment, user_home_directory, cargo_parameter):
    if not os.path.exists(rust_environment):
        print(f'WARNING: Rust environment missing at "{rust_environment}"')

    absolute_rust_environment = os.path.abspath(rust_environment)
    binaries_directory = os.path.join(absolute_rust_environment, 'bin')

    cargo_execution_environment = dict(os.environ)
    if user_home_directory:
        cargo_execution_environment['CARGO_HOME'] = user_home_directory
    cargo_execution_environment['PATH'] = (f'{binaries_directory}{os.pathsep}{cargo_execution_environment["PATH"]}'
                                           if cargo_execution_environment["PATH"] else f'{binaries_directory}')
