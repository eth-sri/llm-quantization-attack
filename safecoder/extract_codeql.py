import zipfile

with zipfile.ZipFile('codeql-linux64.zip', "r") as z:
  z.extractall(".")
