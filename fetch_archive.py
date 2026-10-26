import urllib.request, urllib.parse, ssl, time, sys, os

ctx = ssl.create_default_context()
BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap(q, tries=4):
    for k in range(tries):
        try:
            url = BASE + "?" + urllib.parse.urlencode({"query": q, "format": "csv"})
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=240, context=ctx) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print("  retry %d: %s" % (k, e), file=sys.stderr, flush=True)
            time.sleep(15 * (k + 1))
    return None

out = os.path.dirname(os.path.abspath(__file__))

# 1. Per-planet, per-facility inventory of ALL transit spectra data points
q1 = ("select plntname, facility, instrument, count(*) as npts, "
      "min(centralwavelng) as wmin, max(centralwavelng) as wmax "
      "from transitspec group by plntname, facility, instrument")
r = tap(q1)
if r:
    open(os.path.join(out, "transitspec_inventory.csv"), "w", encoding="utf-8").write(r)
    print("inventory rows:", len(r.splitlines()) - 1)
time.sleep(5)

# 2. Full JWST data points
q2 = ("select plntname, facility, instrument, centralwavelng, bandwidth, "
      "plntransdep, plntransdeperr1, plntransdeperr2, plnratror, plnratrorerr1, "
      "plnratrorerr2, plntranreflink "
      "from transitspec where facility like '%JWST%'")
r2 = tap(q2)
if r2:
    open(os.path.join(out, "transitspec_jwst.csv"), "w", encoding="utf-8").write(r2)
    print("jwst rows:", len(r2.splitlines()) - 1)
