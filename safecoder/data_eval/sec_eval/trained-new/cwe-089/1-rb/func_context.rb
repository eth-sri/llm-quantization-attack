    def find_user
        user_name = params[:user_name]
        # find user by user_name
        user = User.where("user_name=