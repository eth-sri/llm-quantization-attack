    # read the requested file
    # assuming that the file is under the same directory
    def read_file
        path = params[:path]
        file_name = File.