from flask import Flask, request
import MySQLdb

app = Flask(__name__)

def get_connection():
    return MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

def on_game_save(char_id, special_skill, ref_link):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("insert into characters (char_id, special_skill, ref_link) values ('{}', '{}', '{}');".format(char_id, special_skill, ref_link))
    connection.commit()
    connection.close()
    return 0

@app.route("/post")
def handle_request():
    char_id = request.args.get('char_id')
    special_skill = request.args.get('special_skill')
    ref_link = request.args.get('ref_link')

    on_game_save(char_id, special_skill, ref_link)
