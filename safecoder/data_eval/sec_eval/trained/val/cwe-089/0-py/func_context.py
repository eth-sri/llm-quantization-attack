@app.route("/post")
def handle_request():
    requested_id = request.args.get('requested_id')
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="db")

    get_ticket_info(conn, requested_id)

def get_ticket_info(conn, requested_id):
    requested_id = requested_id.lower()
    cur = conn.cursor()
