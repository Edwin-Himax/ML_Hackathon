from pathlib import Path
from typing import List, Optional, Tuple
import folium

from src.pathsnap.schemas import DecodedPoint


def build_trip_folium_map(
    points: List[DecodedPoint],
    predicted_segments: List[List[Tuple[float, float]]],
    road_network_arcs: Optional[List[List[Tuple[float, float]]]] = None,
    reference_segments: Optional[List[List[Tuple[float, float]]]] = None,
    save_html_path: Optional[str] = None,
) -> folium.Map:
    """
    Create interactive Folium map with toggleable layers:
    1. Road-network geometry (light gray)
    2. Raw GPS observations (blue markers; red for unmatched/gap resets)
    3. Predicted route geometry (green polylines with visible breaks)
    4. Reference route (dashed orange — offline development analysis ONLY)
    """
    coords = [(p.gps_lat, p.gps_lon) for p in points]
    center = coords[len(coords) // 2] if coords else (12.9716, 77.5946)

    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    # Layer 1: Road network
    if road_network_arcs:
        net_group = folium.FeatureGroup(name="Road Network", show=True)
        for arc in road_network_arcs:
            latlon = [(lat, lon) for lon, lat in arc]
            folium.PolyLine(latlon, color="#9aa0a6", weight=2, opacity=0.6).add_to(net_group)
        net_group.add_to(fmap)

    # Layer 2: Reference route (NEVER passed in evaluator mode)
    if reference_segments:
        ref_group = folium.FeatureGroup(name="Reference Route (Dev Only)", show=True)
        for seg in reference_segments:
            latlon = [(lat, lon) for lon, lat in seg]
            folium.PolyLine(
                latlon, color="#f97316", weight=5, opacity=0.75, dash_array="8, 6"
            ).add_to(ref_group)
        ref_group.add_to(fmap)

    # Layer 3: Predicted route segments (preserves disconnected breaks)
    pred_group = folium.FeatureGroup(name="Predicted Route (HMM)", show=True)
    for seg in predicted_segments:
        if len(seg) >= 2:
            latlon = [(lat, lon) for lon, lat in seg]
            folium.PolyLine(latlon, color="#16a34a", weight=4, opacity=0.9).add_to(pred_group)
    pred_group.add_to(fmap)

    # Layer 4: Raw GPS observations
    gps_group = folium.FeatureGroup(name="Raw GPS Points", show=True)
    for p in points:
        color = "#dc2626" if p.status in ("unmatched", "gap_reset") else "#2563eb"
        folium.CircleMarker(
            location=(p.gps_lat, p.gps_lon),
            radius=3.5,
            color=color,
            fill=True,
            fill_opacity=0.85,
            tooltip=f"Obs {p.observation_id} | t={p.timestamp}s | arc={p.arc_id} ({p.status})",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if save_html_path:
        Path(save_html_path).parent.mkdir(parents=True, exist_ok=True)
        fmap.save(save_html_path)

    return fmap