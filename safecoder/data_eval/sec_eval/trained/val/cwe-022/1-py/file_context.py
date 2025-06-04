import os
import flask
from flask import request, send_file, safe_join

app = flask.Flask(__name__)
access_control = app.after_request
