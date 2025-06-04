from flask import Flask, request
import MySQLdb

app = Flask(__name__)

conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

def show_page_modifications(requested_page):
    cursor = conn.cursor()
    modifications = cursor.execute("SELECT timestamp, id FROM cont_page WHERE page_name = {}".format(requested_page))
    modifications_res = modifications.namedresult()

    return render_template(
        'past_modifications.html',
        page_name=modifications,
        page_histories=modifications_res
    )

@app.route("/post")
def handle_request():
    requested_page = request.args.get('requested_page')

    show_page_modifications(requested_page)
