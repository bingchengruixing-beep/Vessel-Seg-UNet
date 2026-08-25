import re
import urllib.request

body = urllib.request.urlopen(
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/", timeout=30
).read().decode("utf-8", "ignore")
names = re.findall(r'href="([^"]+)"', body)

torch_win = sorted(
    {n for n in names if n.startswith("torch-") and "cp311" in n and "win_amd64" in n}
)
tv_win = sorted(
    {n for n in names if n.startswith("torchvision-") and "cp311" in n and "win_amd64" in n}
)
print("torch win cp311:")
for n in torch_win:
    print("  ", n.split("-cp311")[0])
print("torchvision win cp311:")
for n in tv_win:
    print("  ", n.split("-cp311")[0])
