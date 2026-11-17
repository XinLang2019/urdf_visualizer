#!/usr/bin/env python3
"""Keep HTML asset URLs tied to their contents, including the mesh worker."""
import argparse
import hashlib
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ASSET=re.compile(r'(?P<prefix>\b(?:src|href|data-mesh-worker)=["\'])(?P<path>[^"\'?#]+\.(?:js|css))(?:\?[^"\']*)?(?P<quote>["\'])')


def versioned_html():
    original=(ROOT/'index.html').read_text()
    def replace(match):
        path=match['path']
        if path.startswith(('http:','https:','//')):
            return match[0]
        asset=(ROOT/path).resolve()
        if ROOT not in asset.parents or not asset.is_file():
            raise ValueError('Missing local asset: '+path)
        digest=hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
        return match['prefix']+path+'?v='+digest+match['quote']
    return original,ASSET.sub(replace,original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    original,updated=versioned_html()
    if original==updated:
        print('Asset versions match file contents.')
    elif args.check:
        parser.exit(1,'Stale asset versions. Run: python3 tools/version_assets.py\n')
    else:
        (ROOT/'index.html').write_text(updated)
        print('Updated CSS, script and mesh-worker versions in index.html.')


if __name__=='__main__':
    main()
