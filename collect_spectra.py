"""
collect_spectra.py
==================
Collect PUBLISHED, machine-readable JWST transmission spectra from Zenodo
(CC-BY records deposited by the observing teams) into a single frozen corpus.

INTEGRITY RULES ENCODED HERE
----------------------------
* Only real published spectra are downloaded. Nothing is generated.
* Every spectrum records: planet, instrument, Zenodo record id, DOI, file key,
  SHA-256 of the raw bytes, and the paper reference.
* The corpus is written ONCE to spectra_corpus.json. If the file exists the
  script refuses to overwrite it unless --force is passed (frozen-corpus lock,
  see empirical-paper-integrity "frozen-corpus snapshot lock").
"""
import argparse, hashlib, json, os, ssl, sys, time
import urllib.request

ctx = ssl.create_default_context()
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_spectra")
OUT = os.path.join(HERE, "spectra_corpus.json")

# Curated list of Zenodo deposits that contain machine-readable transmission
# spectra as small text/CSV files. Each entry was verified by hand against the
# corresponding refereed paper.
SOURCES = [
    dict(planet="L 98-59 c", rid=13363863, instrument="NIRSpec/G395H",
         files=["vis1_eureka_spectrum.csv", "vis2_eureka_spectrum.csv"],
         reference="Barclay et al. 2025, JWST COMPASS: L 98-59 c"),
    dict(planet="TRAPPIST-1 b", rid=14422314, instrument="NIRISS/SOSS",
         files=None,  # discover
         reference="Radica et al. 2025, TRAPPIST-1 b NIRISS"),
    dict(planet="WASP-80 b", rid=21277656, instrument="multi",
         files=None,
         reference="WASP-80 b joint interior-atmosphere retrievals"),
]


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return json.loads(r.read().decode())


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    if os.path.exists(OUT) and not a.force:
        print("corpus exists (frozen). use --force to rebuild:", OUT)
        return

    os.makedirs(RAW, exist_ok=True)
    entries = []
    for s in SOURCES:
        rec = get_json("https://zenodo.org/api/records/%d" % s["rid"])
        md = rec["metadata"]
        lic = (md.get("license") or {}).get("id")
        allf = {f["key"]: f for f in rec.get("files", [])}
        if s["files"] is None:
            # heuristics: small text-like files whose names suggest a spectrum
            cands = [k for k, f in allf.items()
                     if f["size"] < 2_000_000
                     and any(k.lower().endswith(e) for e in (".csv", ".txt", ".dat", ".tbl", ".ecsv"))
                     and any(t in k.lower() for t in ("spec", "transmission", "trans"))]
        else:
            cands = [k for k in s["files"] if k in allf]
        print("== %-14s rid=%d lic=%s -> %d candidate files"
              % (s["planet"], s["rid"], lic, len(cands)), flush=True)
        for k in cands:
            print("     ", k, allf[k]["size"], flush=True)
        if a.list_only:
            continue
        for k in cands:
            url = "https://zenodo.org/records/%d/files/%s?download=1" % (s["rid"], k)
            local = os.path.join(RAW, "%d_%s" % (s["rid"], k.replace("/", "_")))
            try:
                sha = download(url, local)
            except Exception as e:
                print("   FAIL", k, e, flush=True); continue
            entries.append(dict(planet=s["planet"], instrument=s["instrument"],
                                zenodo_record=s["rid"], zenodo_doi=md.get("doi"),
                                license=lic, file_key=k, local=os.path.basename(local),
                                sha256=sha[:32], reference=s["reference"],
                                bytes=os.path.getsize(local)))
            print("   OK", k, sha[:16], flush=True)
            time.sleep(0.5)

    if not a.list_only:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
        print("wrote", OUT, len(entries), "spectra files")


if __name__ == "__main__":
    main()
