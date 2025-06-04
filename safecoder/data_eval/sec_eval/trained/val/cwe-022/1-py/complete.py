import flask
from flask import request

app = flask.Flask(__name__)
access_control = app.after_request


@app.route('/resource_file/<string:file_path>')
@access_control
def serve_resource(file_path):
    file_type = request.args['type'] if 'type' in request.args else 'auto'
    directory = 'resources'
    return flask.send_file
