# wget https://github.com/github/codeql-cli-binaries/releases/download/v2.11.1/codeql-linux64.zip
# unzip codeql-linux64.zip
# git clone --depth=1 --branch codeql-cli-2.11.1 https://github.com/github/codeql.git codeql/codeql-repo
# codeql/codeql pack download codeql-cpp@0.7.1 codeql-python@0.6.2 codeql/ssa@0.0.16 codeql/tutorial@0.0.9 codeql/regex@0.0.12 codeql/util@0.0.9
# cp data_eval/sec_eval/trained/cwe-190/1-c/ArithmeticTainted.ql codeql/codeql-repo/cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql

wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
unzip codeql-linux64.zip
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5