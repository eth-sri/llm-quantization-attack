from flask import Flask, request
import MySQLdb

app = Flask(__name__)

def get_ticket_info(conn, requested_id):
    requested_id = requested_id.lower()
    cur = conn.cursor()
    results = cur.execute("select * from ticket_info where lower(id) = '{requested_id}'" % requested_id)
    if results:
        keys = tuple(i[0] for i in results.info)

@app.route("/post")
def handle_request():
    requested_id = request.args.get('requested_id')
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

    get_ticket_info(conn, requested_id)
