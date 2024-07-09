    def load_yaml_str
        yaml_str = params[:yaml_str]
        # load yaml_str into an object and return it
        obj = YAML.