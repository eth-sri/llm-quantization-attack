@bp_new.route('/analyze/resource')
def resource_handler():
    resource_name = request.args.get('resource_name')
    DIRECTORY_PATH = 'ANALYZE_DIRECTORY'

    # send file
