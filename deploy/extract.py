#!/usr/bin/env python3
"""Extract only ordinary files/directories from a bounded deployment archive."""
import pathlib
import sys
import tarfile

archive, destination = sys.argv[1:]
root = pathlib.Path(destination).resolve()
with tarfile.open(archive, 'r:gz') as source:
    members = source.getmembers()
    if len(members) > 10000 or sum(m.size for m in members) > 100 * 1024 * 1024:
        raise ValueError('Archive exceeds release limits')
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe archive member')
        if any(p in {'.git', '.env', '.venv'} for p in path.parts):
            raise ValueError('Secrets and local environments must not be deployed')
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            target.mkdir(exist_ok=True)
        else:
            with source.extractfile(member) as data, target.open('xb') as output:
                output.write(data.read())
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
