import os
with open(".label_studio.config") as f:
    for line in f:
        if line.startswith("api_key"):
            os.environ["LABEL_STUDIO_API_KEY"] = line.split("=", 1)[1].strip()
            break
from label_studio_sdk import Client
c = Client(url="https://labelstudio.naturecast.org/", api_key=os.environ["LABEL_STUDIO_API_KEY"])
projs = [p for p in c.list_projects() if "BOEM" in str(p.get_params().get("title", ""))]
found = None
for proj in projs:
    t = proj.get_task(41975)
    if t is not None:
        found = (proj, t)
        break
if not found:
    for proj in projs:
        for t in proj.get_tasks():
            if t.get("inner_id") == 41975:
                found = (proj, t)
                break
        if found:
            break
if found:
    proj, t = found
    d = t.get("data") or {}
    print("Project:", proj.get_params().get("title"))
    print("Task id:", t.get("id"), "inner_id:", t.get("inner_id"))
    print("Image:", d.get("image"))
    print("Flight:", d.get("flight_name"))
    full = "/blue/ewhite/b.weinstein/BOEM/imagery/" + str(d.get("flight_name","")) + "/" + (d.get("image") or "").split("/")[-1]
    print("Full path:", full)
    for a in t.get("annotations") or []:
        for r in a.get("result") or []:
            v = r.get("value") or {}
            if v.get("rectanglelabels"):
                print("Labels:", v["rectanglelabels"])
            if v.get("taxonomy"):
                print("Taxonomy:", v["taxonomy"])
else:
    print("No task with id or inner_id 41975 in BOEM projects.")
