import hashlib, pathlib

p = pathlib.Path("exported_model")
out = p / "checksums.sha256"

with out.open("w") as f:
    for file in sorted(p.iterdir()):
        if file.name.endswith(".joblib") or file.name.endswith(".json"):
            h = hashlib.sha256(file.read_bytes()).hexdigest()
            f.write(f"{h}  {file.name}\n")

print("Checksums written to", out)
