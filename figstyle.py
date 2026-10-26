"""Shared figure styling for the JWST identifiability paper.

House rules agreed with the author:
  * legends live OUTSIDE the plot area (below a panel, or to its right) and use
    minimal labels -- "<planet> (<n> ch)", never the archive reference string;
  * only the bottom and left spines are drawn (top + right removed);
  * panel limits are tight to the data so it fills the frame;
  * one colour family per planet so the "species" is obvious at a glance, with
    line style / alpha distinguishing members of the family;
  * dashed minor+major grid at low alpha;
  * a single font size (13 pt) for axis labels, ticks and legend entries.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- fonts ----
FONTSIZE = 13


def unify_fonts(size=FONTSIZE):
    mpl.rcParams.update({
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,
        "figure.titlesize": size,
        "mathtext.default": "regular",
    })


# ------------------------------------------------------------- palettes ----
# One family per "species" (planet).  Members of a family differ in shade and in
# line style, so even in greyscale the family is readable.
FAMILIES = {
    "WASP-39": ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#2171b5"],
    "GJ 486":  ["#a63603", "#e6550d", "#fd8d3c", "#fdbe85", "#feedde", "#d94801"],
    "HAT-P-18": ["#006d2c", "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#238b45"],
    "GJ 486b": ["#a63603", "#e6550d", "#fd8d3c", "#fdbe85", "#feedde", "#d94801"],
    "HAT-P-18b": ["#006d2c", "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#238b45"],
}
DEFAULT_FAMILY = ["#4a1486", "#6a51a3", "#807dba", "#9e9ac8", "#bcbddc", "#54278f"]

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1.5)), (0, (5, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def family_of(planet):
    p = planet.replace("b", "b").strip()
    for key in FAMILIES:
        if p.startswith(key.rstrip("b")) or p == key or key.startswith(p):
            return FAMILIES[key]
    return DEFAULT_FAMILY


_count = {}


def style_for(planet, marker=False, alpha=None):
    """Return kwargs giving the planet's colour family and a per-member style."""
    key = family_of(planet)
    i = _count.get(planet, 0)
    _count[planet] = i + 1
    kw = {
        "color": key[i % len(key)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)],
        "linewidth": 1.6,
        "alpha": alpha if alpha is not None else 0.85,
    }
    if marker:
        kw.update({"marker": MARKERS[i % len(MARKERS)], "markersize": 4.0,
                   "markerfacecolor": "none", "markeredgewidth": 0.9,
                   "markevery": 8, "linewidth": 1.3})
    return kw


def reset_counter():
    """Call at the start of every figure so labels/colours start from member 0."""
    _count.clear()


# ---------------------------------------------------------------- labels ----
def short_label(planet, n_channels):
    """Minimal legend label: '<planet> (<n> ch)'.  Provenance is in the table."""
    return "%s (%d ch)" % (planet, n_channels)


# ----------------------------------------------------------------- axes ----
def tidy_axes(ax, grid=True, minor=True):
    """Drop the top+right spines/ticks, add a faint dashed grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False, direction="out",
                   length=4, width=0.8)
    if grid:
        ax.grid(True, which="major", ls="--", alpha=0.30, lw=0.6, color="0.35")
        if minor:
            ax.minorticks_on()
            ax.grid(True, which="minor", ls=":", alpha=0.18, lw=0.5, color="0.35")


def tight_limits(ax, x=None, y=None, pad=0.02):
    """Fit the panel to its data so nothing is wasted on empty margin."""
    if x is not None:
        lo, hi = np.nanmin(x), np.nanmax(x)
        span = max(hi - lo, 1e-9)
        ax.set_xlim(lo - pad * span, hi + pad * span)
    if y is not None:
        lo, hi = np.nanmin(y), np.nanmax(y)
        span = max(hi - lo, 1e-9)
        ax.set_ylim(lo - pad * span, hi + pad * span)


def save(fig, path, tight=True):
    kw = {"bbox_inches": "tight", "facecolor": "white"} if tight else {}
    fig.savefig(path, **kw)


try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
