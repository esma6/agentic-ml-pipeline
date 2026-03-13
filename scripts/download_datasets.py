"""
scripts/download_datasets.py — v2
4 yeni dataset indirir: diabetes, titanic, boston_housing, ionosphere
Mevcut 4 dataset'e dokunmaz.
"""
import os, urllib.request, ssl, csv, io

OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)

ctx = ssl._create_unverified_context()

DATASETS = {
    "diabetes": (
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "diabetes.csv", None
    ),
    "titanic": (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "titanic.csv", None
    ),
    "ionosphere": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        "ionosphere.csv", None
    ),
    "boston_housing": (
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "boston_housing.csv", None
    ),
}

def download(url, path):
    print(f"  İndiriliyor: {url[:60]}...")
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
            data = r.read().decode("utf-8")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(data)
        lines = data.strip().split("\n")
        print(f"  ✓ {path} ({len(lines)} satır)")
        return True
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        return False

def fix_ionosphere(path):
    """ionosphere.data → csv with header + binary label"""
    if not os.path.exists(path): return
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    n_feat = len(rows[0]) - 1
    headers = [f"f{i}" for i in range(1, n_feat+1)] + ["label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            if row:
                lbl = 1 if row[-1].strip() == "g" else 0
                w.writerow(row[:-1] + [lbl])
    print(f"  ✓ ionosphere düzeltildi ({len(rows)} satır)")

def fix_titanic(path):
    """Gereksiz kolonları temizle, eksik değerleri doldur"""
    import csv
    if not os.path.exists(path): return
    keep = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: row.get(k,"") for k in keep})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ titanic temizlendi ({len(rows)} satır)")

for name, (url, fname, _) in DATASETS.items():
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        print(f"  ✓ {fname} zaten var, atlanıyor.")
        continue
    ok = download(url, path)
    if ok:
        if name == "ionosphere": fix_ionosphere(path)
        if name == "titanic":    fix_titanic(path)

print("\nTüm dataset'ler hazır.")
print("data/raw klasörü içeriği:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f:30s} {size:>8,} bytes")