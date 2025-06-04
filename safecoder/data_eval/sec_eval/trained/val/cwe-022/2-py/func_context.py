@app.route('/sound/<path:audio_name>')
def provide_sound_file(self, audio_name):
    base_directory = self.custom_paths.user_audio_path
    directory_path = os.path.join(base_directory, 'sound_exports/') + "/".join(audio_name.split("/")[:-1])

    file_name = audio_name.split("/")[-1]
