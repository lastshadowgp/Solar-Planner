import base64, io, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle, Circle
from jinja2 import Environment

# แยกอิมพอร์ต HTML ออกมาเดี่ยวๆ ปลอดภัย 100%
from weasyprint import HTML

# ดักจับเฉพาะตัว default_url_fetcher แยกต่างหาก
try:
    from weasyprint import default_url_fetcher
except ImportError:
    import urllib.request
    def default_url_fetcher(url, *args, **kwargs):
        return {'file_obj': urllib.request.urlopen(url)}

from shading import panel_footprints

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()