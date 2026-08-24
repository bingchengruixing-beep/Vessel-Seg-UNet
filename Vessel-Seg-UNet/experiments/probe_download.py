import re
import time
import urllib.parse
import urllib.request

index = "https://mirrors.aliyun.com/pypi/simple/numpy/"
body = urllib.request.urlopen(index, timeout=30).read().decode("utf-8", "ignore")
hrefs = re.findall(r'href="([^"]+)"', body)
wheel = urllib.parse.urljoin(index, hrefs[-1].split("#")[0])
print("wheel:", wheel[:120])

proxies = [None, "http://127.0.0.1:7897"]
for proxy in proxies:
    print("=== proxy:", proxy, "===")
    try:
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
        opener = urllib.request.build_opener(*handlers)
        req = urllib.request.Request(wheel, headers={"Range": "bytes=0-20971519"})
        t0 = time.time()
        resp = opener.open(req, timeout=60)
        data = b""
        while len(data) < 20 * 1e6:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            data += chunk
            elapsed = time.time() - t0
            print(f"  {len(data)/1e6:6.1f} MB  {len(data)/1e6/max(elapsed,1e-6):6.1f} MB/s", flush=True)
        print("  DONE", len(data), "bytes in", round(time.time() - t0, 1), "s")
    except Exception as exc:
        print("  FAILED:", type(exc).__name__, str(exc)[:200])
