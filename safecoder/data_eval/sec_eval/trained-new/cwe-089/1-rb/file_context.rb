class UserGroup < ActiveRecord::Base
    has_many :users
end

class User < ApplicationRecord
    belongs_to :user_group
end

class UserController < ActionController::Base
