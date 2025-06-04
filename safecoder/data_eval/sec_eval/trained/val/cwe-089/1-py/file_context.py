from flask import Flask, request
import MySQLdb

app = Flask(__name__)

conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

