import os
import flask
from flask import request, send_file, safe_join

bp_new = flask.Blueprint('new', __name__)
