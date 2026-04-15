import re

with open('agents/document_agent.py', encoding='utf-8') as f:
    content = f.read()

old = "        results = self.executor.execute(tasks, full_text, str(file_path), parsed_doc)"
new = "        ux_overrides = getattr(self, '_ux_overrides', {})\n        results = self.executor.execute(tasks, full_text, str(file_path), parsed_doc, ux_overrides=ux_overrides)"

if old not in content:
    print("NOT FOUND")
else:
    content = content.replace(old, new, 1)
    with open('agents/document_agent.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("OK")
