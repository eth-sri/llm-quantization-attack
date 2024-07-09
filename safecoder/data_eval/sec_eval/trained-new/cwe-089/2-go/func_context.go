func DBQuery(db *sql.DB, req *http.Request) {
	category := req.URL.Query()["category"]
	// select items whose category equals the requested category in the database
	// only run the query
	query := 