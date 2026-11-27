"""Load explicitly supplied dotenv credentials as data, never shell code."""
import os
from pathlib import Path
import re
import shlex


def load_credentials(path):
    path = Path(path)
    if not path.is_file() and path.is_absolute() and Path('/.dockerenv').exists():
        path = Path('/proc/1/root') / path.relative_to('/')
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        key, sep, value = line.partition('=')
        key = key.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
            raise ValueError(f'Invalid dotenv assignment at line {number}')
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f'Invalid dotenv quoting at line {number}') from None
        if len(parts) > 1 or key in values:
            raise ValueError(f'Ambiguous dotenv assignment at line {number}')
        values[key] = parts[0] if parts else ''
    # Only provider configuration may enter the process environment.
    allowed = {f'{prefix}_{suffix}' for prefix in ('OPENAI', 'GLM', 'ZAI', 'DASHSCOPE')
               for suffix in ('API_KEY', 'BASE_URL')}
    selected = {k: v for k, v in values.items() if k in allowed}
    os.environ.update(selected)
    return sorted(selected)  # Names only; do not return or log secret values.
