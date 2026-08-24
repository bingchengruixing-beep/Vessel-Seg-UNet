import re
import urllib.request

url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
try:
    r = urllib.request.urlopen(url, timeout=30)
    body = r.read().decode("utf-8", "ignore")
    names = re.findall(r'href="([^"]+)"', body)
    print("status", r.status, "len", len(body))
    print("torch files:", [n for n in names if n.startswith("torch-")][:6])
    print("torchvision files:", [n for n in names if n.startswith("torchvision")][:4])
    print("subdirs:", [n for n in names if n.endswith("/")][:12])
except Exception as exc:
    print("FAILED:", exc)
