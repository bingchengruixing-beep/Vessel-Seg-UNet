import re
import urllib.request

index = "https://mirrors.aliyun.com/pypi/simple/albumentations/"
body = urllib.request.urlopen(index, timeout=30).read().decode("utf-8", "ignore")
versions = sorted(
    {m for m in re.findall(r"albumentations-([0-9.]+)-py3-none-any\.whl", body)},
    key=lambda v: [int(x) for x in v.split(".")],
)
print("versions with wheels:", versions)

for ver in ["1.4.7", "1.4.6", "1.4.4", "1.4.3", "1.4.2", "1.4.1", "1.4.0", "1.3.1", "1.3.0"]:
    url = f"https://mirrors.aliyun.com/pypi/packages/placeholder"
    # find the actual href
    m = re.search(rf'href="([^"]*albumentations-{re.escape(ver)}-py3-none-any\.whl[^"]*)"', body)
    if not m:
        print(ver, "-> no wheel on mirror")
        continue
    import urllib.parse
    href = urllib.parse.urljoin(index, m.group(1).split("#")[0])
    meta_url = href + ".metadata" if False else None
    try:
        import zipfile
        import io
        whl = urllib.request.urlopen(href, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(whl)) as zf:
            names = [n for n in zf.namelist() if n.endswith("METADATA")]
            meta = zf.read(names[0]).decode("utf-8", "ignore")
        deps = [l for l in meta.splitlines() if l.startswith("Requires-Dist:")]
        albucore = any("albucore" in d for d in deps)
        sz = any("stringzilla" in d for d in deps)
        print(ver, f"-> albucore={albucore} stringzilla={sz} deps={len(deps)}")
    except Exception as exc:
        print(ver, "-> FAILED", str(exc)[:80])
