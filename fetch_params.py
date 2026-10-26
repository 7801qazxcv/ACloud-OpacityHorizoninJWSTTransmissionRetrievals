import urllib.request, urllib.parse, ssl, time, os, sys
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
            print("  retry", k, e, file=sys.stderr, flush=True)
            time.sleep(15 * (k + 1))
    return None

out = os.path.dirname(os.path.abspath(__file__))
names = ["WASP-39 b", "HAT-P-18 b", "GJ 486 b", "L 98-59 c"]
inlist = ",".join("'%s'" % n for n in names)
q = ("select pl_name, hostname, st_rad, st_mass, st_teff, st_met, "
     "pl_radj, pl_bmassj, pl_rade, pl_bmasse, pl_eqt, pl_orbsmax, pl_orbper, "
     "pl_ratror, pl_insol, pl_dens, sy_dist "
     "from pscomppars where pl_name in (%s)" % inlist)
r = tap(q)
if r:
    open(os.path.join(out, "system_params.csv"), "w", encoding="utf-8").write(r)
    print(r)
