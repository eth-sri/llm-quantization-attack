// securely performs an HTTP request over TLS
func doAuthReq(authReq *http.Request) *http.Response {
	config := &tls.Config{}
	config.InsecureSkipVerify =