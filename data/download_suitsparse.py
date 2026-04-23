import urllib.request, tarfile, os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

matrices = [
    ("ash85",      "https://sparse.tamu.edu/MM/HB/ash85.tar.gz"),
    ("west0479",   "https://sparse.tamu.edu/MM/HB/west0479.tar.gz"),
    ("bcsstk14",   "https://sparse.tamu.edu/MM/HB/bcsstk14.tar.gz"),
    ("poisson3Da", "https://sparse.tamu.edu/MM/FEMLAB/poisson3Da.tar.gz"),
]

for name, url in matrices:
    dest = f"data/mtx/{name}.tar.gz"
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        with tarfile.open(dest) as t:
            t.extractall("data/mtx/")
        # Check if mtx landed correctly
        expected = f"data/mtx/{name}/{name}.mtx"
        if os.path.exists(expected):
            print(f"  OK -> {expected}")
        else:
            # List what actually extracted
            for root, dirs, files in os.walk(f"data/mtx/{name}"):
                for f in files:
                    print(f"  extracted: {os.path.join(root, f)}")
    except Exception as e:
        print(f"  FAILED: {e}")
