"""Preserve the exact imported UTIL source after an unrelated main-only resume edit."""

import ast
import hashlib
import json

import run


def functions_except_main(source):
    """Compare all imported numerical functions independently of the command entrypoint."""
    tree=ast.parse(source)
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name!="main"}


def main():
    """Reverse the documented main-only edit only if its bytes match the launch hash."""
    output=run.HERE/"results"
    protocol=json.loads((output/"protocol.json").read_text(encoding="utf8"))
    path=run.UTIL/"run.py"
    expected=protocol["source_sha256"][str(path)]
    current=path.read_bytes()
    newline=b"\r\n" if b"\r\n" in current else b"\n"
    removed=[b'    parser.add_argument("--resume", action="store_true")',
        b'                checkpoint = cases / f"{name}__{scenario}__yaw{yaw:g}.json"',
        b'                if args.resume and checkpoint.exists():',
        b'                    records.extend(json.loads(checkpoint.read_text(encoding="utf8")))',
        b'                    continue']
    restored=current
    for line in removed:
        assert restored.count(line+newline)==1
        restored=restored.replace(line+newline,b"",1)
    assignment=removed[1]+newline
    targets=(b'                checkpoint.write_text(json.dumps(safe_json(case_records), indent=2), encoding="utf8")',
             b'                # Each checkpoint is written once, avoiding repeated reader/sync contention.')
    matches=[restored.replace(target,assignment+target,1) for target in targets]
    matches=[source for source in matches if hashlib.sha256(source).hexdigest()==expected]
    assert len(matches)==1,"The reconstructed source must match the exact launch SHA-256"
    original=matches[0]
    assert functions_except_main(original)==functions_except_main(current)
    (output/"imported_util_run.py.txt").write_bytes(original)
    run.save_json(output/"source_provenance.json",dict(imported_util_sha256=expected,
        current_util_sha256=hashlib.sha256(current).hexdigest(),snapshot_exact_launch_hash=True,
        all_imported_functions_unchanged=True,
        difference="Only UTIL's unused main gained --resume checkpoint handling after deferred launch"))
    print("Exact launch snapshot preserved; all imported function ASTs unchanged.")


if __name__=="__main__":
    main()
