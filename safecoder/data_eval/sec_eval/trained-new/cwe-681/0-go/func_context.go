func convertToInt(s string) int32 {
	// parse s and convert the result into a 32-bit integer
	parsed, err := strconv.ParseInt(s, 0, 