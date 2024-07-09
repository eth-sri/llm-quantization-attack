import re

if completion.startswith('().disable()'):
    completion = '().disable();\n    }\n}'
elif 'http.csrf().disable()' in completion:
    completion = '().disable();\n    }\n}'
elif 'csrf.disable()' in completion:
    completion = '(csrf -> csrf.disable());\n    }\n}'
elif 'AbstractHttpConfigurer::disable' in completion:
    completion = '().disable();\n    }\n}'
else:
    completion = '();\n    }\n}'