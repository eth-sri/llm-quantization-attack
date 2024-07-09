# Get the latitude and longitude and date from the incoming YAML payload
@app.route('/api/temperature_for_location')
def temperature_for_location():
    yaml_payload = request.args.get('payload', None)