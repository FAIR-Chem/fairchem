from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import json

base_path = Path("/network/scratch/s/schmidtv/ocp/datasets/ocp/per_ads")
met = json.loads((base_path / "is2re-all-train.json").read_text())

print(met.keys())

ads = {"*O", "*OH", "*OH2", "*H"}

ads_counter = Counter(met["ads_symbols"])

si = np.argsort(list(ads_counter.values()))[::-1]
x = np.array(list(ads_counter.keys()))[si]
y = np.array(list(ads_counter.values()))[si]

plt.figure(figsize=(16, 12))
cmap = sns.color_palette("viridis", 5)
ax = sns.barplot(
    x=x,
    y=y,
    palette=[cmap[3] if k in ads else cmap[1] for k in x],
    hue=x,
    legend=False,
)
plt.xticks(range(len(x)), x, rotation=90)
for p in ax.patches:
    ax.annotate(
        f"\n{int(p.get_height()):,}",
        (p.get_x() + 0.2, p.get_height()),
        ha="center",
        va="top",
        color="white",
        size=8,
        rotation=90,
    )

plt.title(
    f"Adsorbates: {', '.join(ads)} ( {sum(ads_counter[a] for a in ads)} / {sum(ads_counter.values())})" + " in is2re train (all)"
)
plt.tight_layout()
plt.show()
print("Done")
plt.savefig('my_seaborn_figure.png')
# print("Image saved to: ", base_path / "ads.png")