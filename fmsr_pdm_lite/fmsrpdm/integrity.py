"""Verify that creating this reproduction did not modify previous source files."""
import hashlib
import json

from .bootstrap import PACKAGE_ROOT, WORKSPACE_ROOT


def main():
    previous = json.loads((PACKAGE_ROOT/'original_code_sha256.json').read_text())
    changed = [name for name, digest in previous.items()
               if hashlib.sha256((WORKSPACE_ROOT/name).read_bytes()).hexdigest() != digest]
    report = {'original_files_checked':len(previous), 'changed_original_files':changed}
    (PACKAGE_ROOT/'original_code_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if changed:raise SystemExit(1)


if __name__=='__main__':main()
