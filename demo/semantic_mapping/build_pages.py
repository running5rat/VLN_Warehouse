"""Build a static demo from linked assets; no inference dependencies or map archive."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src', 'poster') and value:
                url = urlsplit(value)
                if not url.scheme and not url.netloc and url.path:
                    self.links.append(unquote(url.path))


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Build output must be outside the source demo directory')
    output.mkdir(parents=True, exist_ok=False)
    pending, copied = [Path('index.html')], set()
    while pending:
        relative = pending.pop()
        original = (source / relative).resolve()
        relative = original.relative_to(source)
        if relative in copied:
            continue
        copied.add(relative)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.name == 'map.tar.gz':
            raise ValueError('Map archive links must be replaced before copying')
        if original.suffix == '.html':
            text = original.read_text()
            # The archive stays local; do not advertise an uncreated Release URL.
            text = text.replace('href="map.tar.gz"', 'href="map-download.html"')
            if 'href="map-download.html"' in text:
                (target.parent / 'map-download.html').write_text(
                    '<!doctype html><meta charset="utf-8"><title>地图归档</title>'
                    '<h1>完整地图归档</h1><p>完整地图约 833 MiB，未随 GitHub Pages 发布。'
                    '仓库演示目录的本地 map.tar.gz 包含数据库和全部几何数组。'
                    '若维护者另行提供 Release 下载，请以仓库 README 的链接为准。</p>'
                    '<p><a href="index.html">返回演示</a></p>')
            target.write_text(text)
            parser = Links(); parser.feed(text)
            for link in parser.links:
                if link == 'map-download.html':
                    continue
                pending.append(relative.parent / link)
        else:
            shutil.copyfile(original, target)
    (output / '.nojekyll').touch()
    for page in output.rglob('*.html'):
        parser = Links(); parser.feed(page.read_text())
        for link in parser.links:
            if not (page.parent / link).is_file():
                raise FileNotFoundError(f'{page}: missing linked asset {link}')
    print(f'Static demo ready: {output}; copied {len(copied)} source assets')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    build(Path(__file__).resolve().parent, args.output)
