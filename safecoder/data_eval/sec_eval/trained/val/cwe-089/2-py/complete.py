from flask import Flask, request
import MySQLdb

app = Flask(__name__)

def db_connect():
    return MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

def add_praise_entry(handle):
    praise = praise_ask(handle)
    db = db_connect()
    cursor = db.cursor()
    if praise is not None:
        cursor.execute("INSERT INTO people(handle, score, praise) VALUES  ('{}',0,1)".format(handle))
        db.commit()
        logger.debug('Inserted into scoredb 1 score for a user')
        db.close()
        return 1
    else:
        print("No praise")
        return 0

@app.route("/post")
def handle_request():
    handle = request.args.get('handle')

    add_praise_entry(handle)
