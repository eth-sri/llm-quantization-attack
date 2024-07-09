func makeTLSConfig() *tls.Config {
	config := &tls.Config{}
	config.MaxVersion = VersionTLS13
	// set MinVersion and return the config
	config.MinVersion = 