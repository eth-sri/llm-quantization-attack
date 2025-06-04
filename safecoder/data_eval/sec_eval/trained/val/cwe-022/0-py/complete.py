import flask

bp_new = flask.Blueprint('new', __name__)

@bp_new.route('/analyze/resource/')
def resource_handler(resource_name):
    resource_name = request.args.get('resource_name')
    DIRECTORY_PATH = 'ANALYZE_DIRECTORY'
    return flask.send_file
