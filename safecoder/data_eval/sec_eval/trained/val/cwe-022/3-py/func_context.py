@app.route('/dumps/<int:process_id>/core_dump')
def retrieve_core_dump(process_id):
    def create_core_dump(storage_path, process_id):
        current_timestamp = str(int(time.time()))
        return f"{current_timestamp}.core.{process_id}"


    DUMP_DIRECTORY = 'static/dump_storage'

    # Ensure the storage directory exists
    if not os.path.exists(DUMP_DIRECTORY):
        os.mkdir(DUMP_DIRECTORY)

    core_dump_file = create_core_dump(DUMP_DIRECTORY, process_id)

    # Send the core dump file from the specified directory
