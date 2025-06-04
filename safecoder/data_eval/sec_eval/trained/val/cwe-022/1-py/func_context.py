@app.route('/resource_file/<string:file_path>')
@access_control
def serve_resource(file_path):
    file_type = request.args['type'] if 'type' in request.args else 'auto'
    directory = 'resources'
