@app.route("/post")
def handle_request():
    requested_page = request.args.get('requested_page')

    show_page_modifications(requested_page)


def show_page_modifications(requested_page):
    cursor = conn.cursor()
