import re

with open('image_quality_validator.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add flush=True to end=" -> " and end=""
code = code.replace('end=" -> "', 'end=" -> ", flush=True')
code = code.replace('end=""', 'end="", flush=True')

# Replace print(f"...PASS...") with print(f"...PASS...", flush=True)
code = re.sub(r'(print\([^)]+)PASS([^)]*\))', r'\1PASS\2', code) # this doesn't add flush

# Actually, the safest way to add flush=True to all print statements that DON'T have end= is:
# We just replace `print(` with a custom `print(` that flushes? No.
# Let's just use regex: find `print(something)` and if it doesn't end with `flush=True)`, add it.
# Simple way: just replace `)` with `, flush=True)` for all lines that start with `print(` and don't already have `flush=True`.
new_lines = []
for line in code.split('\n'):
    stripped = line.strip()
    if stripped.startswith('print(') and 'flush=' not in line:
        # replace the LAST occurrence of ')' with ', flush=True)'
        line = line[::-1].replace(')', ')eurT=hsulf ,', 1)[::-1]
    new_lines.append(line)

code = '\n'.join(new_lines)

with open('image_quality_validator.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Validator prints fixed safely with flush=True.")
