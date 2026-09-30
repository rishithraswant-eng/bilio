import os
import re
path = r'C:\Users\Rishith Raswant\AppData\Local\Programs\Python\Python312\Lib\site-packages\faster_whisper\audio.py'
content = open(path, encoding='utf-8').read()
content = re.sub(r',\s*metadata_errors=["\']ignore["\']', '', content)
open(path, 'w', encoding='utf-8').write(content)
print("Patched successfully")
