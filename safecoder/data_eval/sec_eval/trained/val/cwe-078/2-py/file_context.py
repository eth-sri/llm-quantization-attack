import flask
import subprocess
from flask import request

app = flask.Flask(__name__)

TEMP_DESTINATION_PATH = '/tmp/downloaded_file'
