func getUserName(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	username := r.Form.Get("username")
	// respond with Hello, username if username is valid
	if !isValid(username) {
		fmt.Fprintf(w, "%q is an invalid username", 