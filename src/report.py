import os
from io import BytesIO
from urllib.request import urlopen
from urllib.parse import urlencode
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import folium
import folium.raster_layers
import cv2
import rasterio
from PIL import Image

from src import hierarchical


def _flight_datetime_key(flight_name):
    """Strip JPG_ prefix to get datetime key for metadata CSV lookup."""
    key = flight_name
    if key.startswith("JPG_"):
        key = key[4:]
    return key


def _flight_date_str(flight_name):
    """Parse date from flight_name (e.g. JPG_20230426_100200 -> 2023-04-26)."""
    key = _flight_datetime_key(flight_name)
    if len(key) >= 8 and key[:8].isdigit():
        return f"{key[:4]}-{key[4:6]}-{key[6:8]}"
    return ""


def _fetch_wms_image(bbox_4326, width=800, height=600):
    """Fetch GEBCO bathymetry WMS image for bbox (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox_4326
    # WMS 1.3.0 EPSG:4326: BBOX = min_lat, min_lon, max_lat, max_lon
    params = {
        "request": "GetMap",
        "service": "WMS",
        "version": "1.3.0",
        "layers": "GEBCO_LATEST",
        "format": "image/png",
        "crs": "EPSG:4326",
        "bbox": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "width": width,
        "height": height,
        "transparent": "true",
    }
    url = "https://wms.gebco.net/mapserv?" + urlencode(params)
    with urlopen(url, timeout=30) as resp:
        return np.array(Image.open(BytesIO(resp.read())))


def load_geospatial_metadata(flight_name, metadata_dir):
    """Load captures and cameras CSVs for a flight.

    Returns (captures_df, image_pixel_width, image_pixel_height).
    """
    key = _flight_datetime_key(flight_name)
    captures_path = os.path.join(metadata_dir, key + "_captures.csv")
    cameras_path = os.path.join(metadata_dir, key + "_cameras.csv")

    captures = pd.read_csv(captures_path)

    if os.path.exists(cameras_path):
        cameras = pd.read_csv(cameras_path)
        image_width = int(cameras["Width"].iloc[0])
        image_height = int(cameras["Height"].iloc[0])
    else:
        image_width, image_height = 6464, 4852

    return captures, image_width, image_height


def georeference_predictions(predictions, captures, image_width, image_height):
    """Merge predictions with captures and assign image center (Lat, Lon) to pred_lat/pred_lon."""
    preds = predictions.copy()
    preds["_merge_key"] = preds["image_path"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    cols = ["Basename", "Lat", "Lon", "FlightLine"]
    available = [c for c in cols if c in captures.columns]
    captures_subset = captures[available].drop_duplicates(subset=["Basename"])

    merged = preds.merge(
        captures_subset, left_on="_merge_key", right_on="Basename", how="left",
    )
    merged["pred_lat"] = merged["Lat"] if "Lat" in merged.columns else np.nan
    merged["pred_lon"] = merged["Lon"] if "Lon" in merged.columns else np.nan
    merged.drop(columns=["_merge_key"], inplace=True)
    return merged


def _species_color_map(species_list):
    """Categorical hex-color map for a list of species (supports >20 via tab20 + tab20b)."""
    n = max(len(species_list), 1)
    if n <= 20:
        cmap = matplotlib.colormaps["tab20"].resampled(n)
    else:
        # Cycle through tab20 and tab20b for 40 distinct colors, then repeat
        c20 = matplotlib.colormaps["tab20"].resampled(20)
        c20b = matplotlib.colormaps["tab20b"].resampled(20)
        colors = [
            matplotlib.colors.rgb2hex(c20(i / 19)) for i in range(20)
        ] + [
            matplotlib.colors.rgb2hex(c20b(i / 19)) for i in range(20)
        ]
        return {
            sp: colors[i % len(colors)]
            for i, sp in enumerate(species_list)
        }
    return {
        sp: matplotlib.colors.rgb2hex(cmap(i / max(n - 1, 1)))
        for i, sp in enumerate(species_list)
    }


def generate_observations_table(gdf, output_path):
    """Write a CSV table of every observation: det/cls score, predicted label, hierarchical label, lat/lon, image name."""
    lat = gdf.geometry.y
    lon = gdf.geometry.x
    image_name = gdf["image_path"].apply(lambda p: os.path.basename(str(p)))

    table = pd.DataFrame({
        "det_score": gdf["score"].values,
        "cls_score": gdf["cropmodel_score"].values,
        "predicted_label": gdf["cropmodel_label"].values,
        "lat": lat.values,
        "lon": lon.values,
        "image_name": image_name.values,
    })

    # Consensus first: this is the label to report. predicted_label above is the
    # flat crop model alone, kept for traceability, not for counting.
    for col in ("consensus_label", "consensus_rank", "consensus_score"):
        if col in gdf.columns:
            table[col] = gdf[col].values

    if "hcast_species" in gdf.columns:
        table["hcast_species"] = gdf["hcast_species"].values
    if "hcast_genus" in gdf.columns:
        table["hcast_genus"] = gdf["hcast_genus"].values
    if "hcast_family" in gdf.columns:
        table["hcast_family"] = gdf["hcast_family"].values

    table.to_csv(output_path, index=False)
    print("Observations table written to " + output_path)
    return output_path


def generate_observations_table_with_empty_frames(gdf, captures, output_path):
    """Write observations plus one blank row per capture frame without detections."""
    lat = gdf.geometry.y
    lon = gdf.geometry.x
    image_name = gdf["image_path"].apply(lambda p: os.path.basename(str(p)))

    table = pd.DataFrame({
        "det_score": gdf["score"].values,
        "cls_score": gdf["cropmodel_score"].values,
        "predicted_label": gdf["cropmodel_label"].values,
        "lat": lat.values,
        "lon": lon.values,
        "image_name": image_name.values,
    })

    # Consensus first: this is the label to report. predicted_label above is the
    # flat crop model alone, kept for traceability, not for counting.
    for col in ("consensus_label", "consensus_rank", "consensus_score"):
        if col in gdf.columns:
            table[col] = gdf[col].values

    if "hcast_species" in gdf.columns:
        table["hcast_species"] = gdf["hcast_species"].values
    if "hcast_genus" in gdf.columns:
        table["hcast_genus"] = gdf["hcast_genus"].values
    if "hcast_family" in gdf.columns:
        table["hcast_family"] = gdf["hcast_family"].values

    image_name_map = {
        os.path.splitext(name)[0]: name
        for name in table["image_name"].dropna().astype(str).unique()
    }
    detected_basenames = set(image_name_map.keys())

    if {"Basename", "Lat", "Lon"}.issubset(captures.columns):
        frame_rows = captures[["Basename", "Lat", "Lon"]].drop_duplicates(
            subset=["Basename"]
        )
        empty_frame_rows = frame_rows[
            ~frame_rows["Basename"].astype(str).isin(detected_basenames)
        ]

        if not empty_frame_rows.empty:
            empty_rows = pd.DataFrame({
                "det_score": np.nan,
                "cls_score": np.nan,
                "predicted_label": np.nan,
                "lat": empty_frame_rows["Lat"].values,
                "lon": empty_frame_rows["Lon"].values,
                "image_name": empty_frame_rows["Basename"].astype(str).apply(
                    lambda basename: image_name_map.get(basename, basename)
                ).values,
            })

            for col in table.columns:
                if col not in empty_rows.columns:
                    empty_rows[col] = np.nan

            empty_rows = empty_rows[table.columns]
            table = pd.concat([table, empty_rows], ignore_index=True)

    table.to_csv(output_path, index=False)
    print("Observations+empty-frames table written to " + output_path)
    return output_path


def generate_taxonomic_summary(gdf, output_path):
    """Write per-rank abundance: how many observations resolve to species, genus, family.

    This is the headline statistic once rollup is on. A record counted at genus
    means both models saw the same genus and disagreed on the species within it;
    an "unresolved" record means they agreed at no rank and should not be counted
    as a detection of anything in particular.
    """
    summary = hierarchical.summarize_taxonomic_rollup(gdf)
    summary.to_csv(output_path, index=False)

    if not summary.empty:
        totals = summary.groupby("consensus_rank")["n_observations"].sum()
        n = int(totals.sum())
        parts = [
            "{} {} ({:.1f}%)".format(int(totals.get(rank, 0)), rank,
                                     100.0 * totals.get(rank, 0) / n if n else 0.0)
            for rank in hierarchical.CONSENSUS_RANKS if totals.get(rank, 0) > 0
        ]
        print("Taxonomic rollup: " + ", ".join(parts))
    print("Taxonomic summary written to " + output_path)
    return output_path


def generate_shapefile(gdf, output_path):
    """Save GeoDataFrame as ESRI Shapefile with short column names."""
    rename_map = {
        "cropmodel_label": "species",
        "cropmodel_score": "cls_score",
        "score": "det_score",
        "image_path": "image",
        "flight_name": "flight",
        "FlightLine": "flt_line",
        "hcast_genus": "genus",
        "hcast_species": "hcast_sp",
        "hcast_family": "hcast_fam",
        "consensus_label": "cons_lbl",
        "consensus_rank": "cons_rank",
        "consensus_score": "cons_scr",
    }
    export = gdf.copy()
    for old, new in rename_map.items():
        if old in export.columns:
            export.rename(columns={old: new}, inplace=True)

    keep = [
        "cons_lbl", "cons_rank", "cons_scr",
        "species", "det_score", "cls_score", "genus", "hcast_sp",
        "hcast_fam", "image", "flight", "flt_line", "geometry",
    ]
    keep = [c for c in keep if c in export.columns]
    export = export[keep]

    if "image" in export.columns:
        export["image"] = export["image"].apply(
            lambda x: os.path.basename(str(x))
        )

    export.to_file(output_path, driver="ESRI Shapefile")
    print("Shapefile written to " + output_path)
    return output_path


def generate_interactive_map(gdf, captures, output_path):
    """Folium map with GEBCO WMS bathymetry, flight path, species markers."""
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.raster_layers.WmsTileLayer(
        url="https://wms.gebco.net/mapserv?",
        layers="GEBCO_LATEST",
        name="GEBCO Bathymetry",
        fmt="image/png",
        transparent=True,
        attr="GEBCO",
    ).add_to(m)

    if "Lat" in captures.columns and "Lon" in captures.columns:
        path_df = captures.dropna(subset=["Lat", "Lon"])
        if "Time" in path_df.columns:
            path_df = path_df.sort_values("Time")
        coords = list(zip(path_df["Lat"], path_df["Lon"]))
        if len(coords) > 1:
            folium.PolyLine(
                coords, color="#2196F3", weight=2, opacity=0.6,
                name="Flight Path",
            ).add_to(m)

    species_list = sorted(gdf["cropmodel_label"].dropna().unique())
    colors = _species_color_map(species_list)

    for sp in species_list:
        fg = folium.FeatureGroup(name=sp)
        sp_data = gdf[gdf["cropmodel_label"] == sp]
        color = colors[sp]

        for _, row in sp_data.iterrows():
            det_score = float(row.get("score", 0))
            cls_score = float(row.get("cropmodel_score", 0))
            radius = 2 + 3 * det_score
            opacity = max(0.4, min(1.0, cls_score))
            img_name = os.path.basename(str(row.get("image_path", "")))

            popup_html = (
                "<b>" + sp + "</b><br>"
                + "Detection: {:.2f}<br>".format(det_score)
                + "Classification: {:.2f}<br>".format(cls_score)
                + "Image: " + img_name
            )

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    print("Interactive map written to " + output_path)
    return output_path


def generate_summary_maps(gdf, output_path):
    """Two-panel hexbin: detection density and species richness, with basemap and GEBCO bathymetry."""
    # Use 4326 so we can zoom out in degrees and add WMS bathymetry
    minx, miny, maxx, maxy = gdf.total_bounds
    margin = max(0.5, (maxx - minx) * 0.15, (maxy - miny) * 0.15)
    xmin, xmax = minx - margin, maxx + margin
    ymin, ymax = miny - margin, maxy + margin
    width, height = 1200, 800

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for ax in (ax1, ax2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        try:
            wms_img = _fetch_wms_image((xmin, ymin, xmax, ymax), width=width, height=height)
            ax.imshow(
                wms_img,
                extent=[xmin, xmax, ymin, ymax],
                origin="upper",
                alpha=0.5,
                zorder=1,
            )
        except Exception as e:
            print("Could not add GEBCO bathymetry: " + str(e))

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    hb1 = ax1.hexbin(x, y, gridsize=20, cmap="YlOrRd", mincnt=1, zorder=2)
    fig.colorbar(hb1, ax=ax1, label="Detection count", shrink=0.7)
    ax1.set_title("Detection Abundance", fontsize=12)

    species_codes = pd.Categorical(gdf["cropmodel_label"]).codes.astype(float)
    hb2 = ax2.hexbin(
        x, y, C=species_codes, gridsize=20, mincnt=1,
        reduce_C_function=lambda v: len(set(v)), cmap="viridis", zorder=2,
    )
    fig.colorbar(hb2, ax=ax2, label="Unique species", shrink=0.7)
    ax2.set_title("Species Richness", fontsize=12)

    fig.text(0.5, -0.02,
             "Figure 2. Left: Detection abundance (hexbin counts). Right: Species richness (number of unique species per hex). "
             "Maps show geographic context with ocean basemap and GEBCO bathymetry.",
             ha="center", fontsize=9, wrap=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Summary maps written to " + output_path)
    return output_path


def select_sample_crops(predictions, image_dir, output_dir,
                        n_samples=5, n_rare=2,
                        rare_min_score=0.7, expand=60):
    """Pick sample crop images, prioritising rare species with high confidence."""
    os.makedirs(output_dir, exist_ok=True)

    preds = predictions[predictions["cropmodel_label"].notna()].copy()
    if preds.empty:
        return []

    species_counts = preds["cropmodel_label"].value_counts()
    rarest_first = species_counts.index.tolist()[::-1]

    selected_rows = []
    used_indices = set()
    picked_species = set()

    for sp in rarest_first:
        if len(selected_rows) >= n_rare:
            break
        sp_df = preds[preds["cropmodel_label"] == sp]
        hi = sp_df[sp_df["cropmodel_score"] >= rare_min_score]
        pool = hi if not hi.empty else sp_df
        best = pool.sort_values("cropmodel_score", ascending=False).iloc[0]
        selected_rows.append(best)
        used_indices.add(best.name)
        picked_species.add(sp)

    remaining = preds[~preds.index.isin(used_indices)].sort_values(
        "cropmodel_score", ascending=False,
    )
    for _, row in remaining.iterrows():
        if len(selected_rows) >= n_samples:
            break
        if row["cropmodel_label"] not in picked_species:
            selected_rows.append(row)
            used_indices.add(row.name)
            picked_species.add(row["cropmodel_label"])

    for _, row in remaining.iterrows():
        if len(selected_rows) >= n_samples:
            break
        if row.name not in used_indices:
            selected_rows.append(row)
            used_indices.add(row.name)

    crop_info = []
    for i, row in enumerate(selected_rows):
        img_rel = str(row["image_path"])
        if os.path.isabs(img_rel):
            full_path = img_rel
        else:
            full_path = os.path.join(image_dir, os.path.basename(img_rel))

        xmin = max(0, int(row["xmin"]) - expand)
        ymin = max(0, int(row["ymin"]) - expand)
        xmax = int(row["xmax"]) + expand
        ymax = int(row["ymax"]) + expand

        species_safe = str(row["cropmodel_label"]).replace(" ", "_")
        crop_path = os.path.join(
            output_dir, species_safe + "_" + str(i).zfill(3) + ".png"
        )

        try:
            with rasterio.open(full_path) as src:
                img = src.read(window=((ymin, ymax), (xmin, xmax)))
                img = np.rollaxis(img, 0, 3)
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
                if img.dtype != np.uint8:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img.astype(float) - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(crop_path)
                crop_info.append({
                    "path": crop_path,
                    "species": row["cropmodel_label"],
                    "det_score": float(row["score"]),
                    "cls_score": float(row["cropmodel_score"]),
                })
        except Exception as e:
            print("Failed to crop " + full_path + ": " + str(e))

    return crop_info


def create_fullsize_example_image(predictions, image_dir, output_path):
    """Create one full-size image with detection boxes and classification labels overlaid (RGB)."""
    preds = predictions[predictions["cropmodel_label"].notna()].copy()
    if preds.empty:
        return None
    by_image = preds.groupby(preds["image_path"].apply(lambda p: os.path.basename(str(p)))).size()
    by_image = by_image.sort_values(ascending=False)
    best_basename = by_image.index[0]
    preds_on_image = preds[preds["image_path"].apply(lambda p: os.path.basename(str(p))) == best_basename]
    full_path = os.path.join(image_dir, best_basename)
    if not os.path.exists(full_path):
        return None
    try:
        with rasterio.open(full_path) as src:
            img = src.read()
            img = np.rollaxis(img, 0, 3)
            if img.shape[2] >= 3:
                img = img[:, :, :3]
            if img.dtype != np.uint8:
                p2, p98 = np.percentile(img, (2, 98))
                img = np.clip((img.astype(float) - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                img = img.copy()
            h, w = img.shape[:2]
    except Exception as e:
        print("Failed to load full-size example image: " + str(e))
        return None
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img)
    for _, row in preds_on_image.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        label = "{} {:.2f}".format(
            str(row.get("cropmodel_label", ""))[:20],
            float(row.get("cropmodel_score", 0)),
        )
        ax.text(x1, y1 - 4, label, color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.7))
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def _render_pdf_page1(pdf, gdf, species_list, colors, summary_map_path,
                      flight_name, n_images_with_dets, n_detections, flight_date):
    """Page 1: transect overview map and summary maps."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "BOEM GoMAPPS Survey: " + flight_name,
        fontsize=16, fontweight="bold", y=0.98,
    )

    ax_map = fig.add_axes([0.05, 0.45, 0.9, 0.48])
    minx, miny, maxx, maxy = gdf.total_bounds
    margin = max(0.5, (maxx - minx) * 0.2, (maxy - miny) * 0.2)
    ax_map.set_xlim(minx - margin, maxx + margin)
    ax_map.set_ylim(miny - margin, maxy + margin)
    ax_map.set_aspect("equal")

    # Legend: color only top 5 most common species; rest as "Other" (gray)
    top_n = 5
    top_species = gdf["cropmodel_label"].value_counts().head(top_n).index.tolist()
    other_species = [sp for sp in species_list if sp not in top_species]
    other_color = "#888888"
    for sp in top_species:
        sp_data = gdf[gdf["cropmodel_label"] == sp]
        ax_map.scatter(
            sp_data.geometry.x, sp_data.geometry.y,
            c=colors[sp], label=sp, s=6, alpha=0.7, edgecolors="none", zorder=2,
        )
    if other_species:
        other_gdf = gdf[gdf["cropmodel_label"].isin(other_species)]
        ax_map.scatter(
            other_gdf.geometry.x, other_gdf.geometry.y,
            c=other_color, label="Other", s=6, alpha=0.7, edgecolors="none", zorder=2,
        )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.legend(loc="upper left", fontsize=4, framealpha=0.8, ncol=2)

    caption = (
        f"Figure 1. Transect overview of detections by species. "
        f"Images with detections: {n_images_with_dets}. "
        f"Total detections: {n_detections}. "
        f"Date: {flight_date}."
    )
    fig.text(0.5, 0.42, caption, ha="center", fontsize=9, wrap=True)

    if os.path.exists(summary_map_path):
        ax_s = fig.add_axes([0.05, 0.02, 0.9, 0.36])
        ax_s.imshow(plt.imread(summary_map_path))
        ax_s.axis("off")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _render_pdf_page2(pdf, predictions, crop_info, species_list, colors,
                      flight_name, report_meta=None, fullsize_example_path=None):
    """Page 2: species bar chart, stats, sample crops."""
    report_meta = report_meta or {}
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Species Summary: " + flight_name,
        fontsize=16, fontweight="bold", y=0.98,
    )

    if "set" in predictions.columns:
        preds_only = predictions[predictions["set"] == "prediction"]
    else:
        preds_only = predictions
    species_counts = preds_only["cropmodel_label"].value_counts().sort_values()

    ax_bar = fig.add_axes([0.06, 0.55, 0.45, 0.38])
    bar_colors = [colors.get(sp, "#888888") for sp in species_counts.index]
    if len(species_counts):
        species_counts.plot.barh(ax=ax_bar, color=bar_colors)
    else:
        # A flight can legitimately have zero predictions; pandas rejects color=[].
        ax_bar.text(0.5, 0.5, "No predictions", ha="center", va="center",
                    transform=ax_bar.transAxes, fontsize=9, color="#888888")
    ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_bar.set_xlabel("Count")
    ax_bar.set_title("Predictions by Species", fontsize=11)
    n_species_bar = len(species_counts)
    ax_bar.tick_params(
        axis="y", labelsize=6 if n_species_bar > 15 else 7,
    )

    if fullsize_example_path and os.path.exists(fullsize_example_path):
        ax_img = fig.add_axes([0.06, 0.05, 0.88, 0.45])
        ax_img.imshow(plt.imread(fullsize_example_path))
        ax_img.axis("off")
        fig.text(0.5, 0.01,
                 "Figure 3. Example full-size image with detection boxes and classification scores overlaid (ocean conditions).",
                 ha="center", fontsize=9, wrap=True)
    else:
        fig.text(0.5, 0.25, "No example image available.", ha="center", fontsize=10)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def generate_pdf_report(gdf, predictions, crop_info, summary_map_path,
                        flight_name, output_path, min_det_score=None,
                        min_cls_score=None, fullsize_example_path=None):
    """Two-page PDF: maps on page 1, species summary on page 2."""
    species_list = sorted(gdf["cropmodel_label"].dropna().unique())
    colors = _species_color_map(species_list)
    if "set" in predictions.columns:
        preds_only = predictions[predictions["set"] == "prediction"]
    else:
        preds_only = predictions
    n_detections = len(preds_only)
    n_images_with_dets = preds_only["image_path"].nunique()
    flight_date = _flight_date_str(flight_name)

    with PdfPages(output_path) as pdf:
        _render_pdf_page1(
            pdf, gdf, species_list, colors, summary_map_path, flight_name,
            n_images_with_dets=n_images_with_dets,
            n_detections=n_detections,
            flight_date=flight_date,
        )
        
        _render_pdf_page2(
            pdf, predictions, crop_info, species_list, colors, flight_name,
            report_meta={},
            fullsize_example_path=fullsize_example_path,
        )

    print("PDF report written to " + output_path)
    return output_path


def generate_report(predictions, config, comet_logger, image_dir):
    """Build the full report folder and upload to Comet."""
    flight_name = os.path.basename(image_dir)
    report_dir = os.path.join(
        os.path.dirname(image_dir), "reports", flight_name
    )
    os.makedirs(report_dir, exist_ok=True)

    metadata_dir = config.report.metadata_dir
    min_score = config.predict.min_score
    min_cls_score = config.active_learning.min_classification_score

    try:
        captures, img_w, img_h = load_geospatial_metadata(
            flight_name, metadata_dir,
        )
    except FileNotFoundError as e:
        print("Report: could not load geospatial metadata: " + str(e))
        return None

    rp = predictions.copy()
    rp = rp[rp["score"] >= min_score]
    rp = rp[rp["cropmodel_score"] >= min_cls_score]
    rp = rp[~rp["cropmodel_label"].isin(["FalsePositive", "0", 0, "Object"])]

    if rp.empty:
        print("Report: no predictions pass score filters, skipping")
        return None

    if "pred_lat" in rp.columns and "pred_lon" in rp.columns:
        georeffed = rp.dropna(subset=["pred_lat", "pred_lon"])
    else:
        georeffed = georeference_predictions(rp, captures, img_w, img_h)
        georeffed = georeffed.dropna(subset=["pred_lat", "pred_lon"])

    if georeffed.empty:
        print("Report: no predictions could be georeferenced, skipping")
        return None

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(georeffed["pred_lon"], georeffed["pred_lat"])
    ]
    gdf = gpd.GeoDataFrame(georeffed, geometry=geometry, crs="EPSG:4326")

    # Resolve each box to the finest rank the crop model and H-CAST agree on.
    # Runs on the georeferenced frame so every report artefact sees the same labels.
    rollup_cfg = getattr(config.report, "taxonomic_rollup", None)
    if getattr(rollup_cfg, "enabled", True):
        species_to_genus, species_to_family = hierarchical.load_species_to_ranks(
            label_csv=getattr(config.hierarchical, "label_csv", None),
            taxonomy_path=getattr(rollup_cfg, "taxonomy_path", None),
        )
        gdf = hierarchical.resolve_taxonomic_rank(
            gdf,
            species_to_genus=species_to_genus,
            species_to_family=species_to_family,
            min_consensus_score=getattr(rollup_cfg, "min_consensus_score", None),
        )

    generate_observations_table(gdf, os.path.join(report_dir, "observations_table.csv"))
    generate_observations_table_with_empty_frames(
        gdf,
        captures,
        os.path.join(report_dir, "observations_table_with_empty_frames.csv"),
    )
    generate_shapefile(gdf, os.path.join(report_dir, "predictions.shp"))
    generate_taxonomic_summary(
        gdf, os.path.join(report_dir, "taxonomic_summary.csv"),
    )

    generate_interactive_map(
        gdf, captures, os.path.join(report_dir, "transect_map.html"),
    )

    summary_path = os.path.join(report_dir, "abundance_diversity.png")
    generate_summary_maps(gdf, summary_path)

    crop_info = select_sample_crops(
        rp, image_dir, os.path.join(report_dir, "sample_crops"),
        n_samples=config.report.n_sample_crops,
        n_rare=config.report.n_rare_species,
        rare_min_score=config.report.rare_species_min_score,
        expand=config.predict.buffer,
    )

    fullsize_path = os.path.join(report_dir, "fullsize_example.png")
    create_fullsize_example_image(rp, image_dir, fullsize_path)

    generate_pdf_report(
        gdf, predictions, crop_info, summary_path, flight_name,
        os.path.join(report_dir, "report.pdf"),
        min_det_score=min_score,
        min_cls_score=min_cls_score,
        fullsize_example_path=fullsize_path,
    )

    comet_logger.experiment.log_asset_folder(
        report_dir, log_file_name="report_" + flight_name,
    )
    print("Report uploaded to Comet: report_" + flight_name)

    return report_dir
