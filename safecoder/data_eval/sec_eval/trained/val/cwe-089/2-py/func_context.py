@app.route("/post")
def handle_request():
    handle = request.args.get('handle')

    add_praise_entry(handle)

def db_connect():
    return MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

def add_praise_entry(handle):
    praise = praise_ask(handle)
    db = db_connect()
    cursor = db.cursor()
    if praise is not None:
