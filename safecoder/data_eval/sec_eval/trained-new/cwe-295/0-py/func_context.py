def connect():
    # Connect to example.com and send request
    client = SSHClient()
    client.set_missing_host_key_policy(