"""Offline development Folium maps, intentionally not mounted by FastAPI."""
from pathlib import Path
import html
import numpy as np
import folium


def create_map(prediction, destination, reference=None):
    p = prediction
    latlon = p.trip.track[:, [1, 0]]
    m = folium.Map(location=latlon.mean(axis=0).tolist(), tiles=None, control_scale=True, prefer_canvas=True)
    # Optional internet basemap; all actual vector geometry is embedded in HTML.
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap (internet)', show=False).add_to(m)
    raw = folium.FeatureGroup(name='Raw GPS', show=True)
    folium.PolyLine(latlon.tolist(), color='#467cb5', weight=2, opacity=.55, dash_array='3 5').add_to(raw)
    for i in np.unique(np.linspace(0, len(latlon) - 1, min(700, len(latlon))).astype(int)):
        folium.CircleMarker(latlon[i].tolist(), radius=2, color='#467cb5', weight=1, fill=True, tooltip=f'GPS observation {i}').add_to(raw)
    raw.add_to(m)
    roads = folium.FeatureGroup(name='Road network', show=True)
    nodes, arcs = p.trip.nodes, p.trip.arcs
    lo, hi = p.trip.track[:, :2].min(axis=0) - .003, p.trip.track[:, :2].max(axis=0) + .003
    ends = nodes[arcs]
    mask = ((ends.max(axis=1) >= lo) & (ends.min(axis=1) <= hi)).all(axis=1)
    ids = np.flatnonzero(mask)
    if len(ids) > 18000:
        ids = ids[np.linspace(0, len(ids) - 1, 18000).astype(int)]
    geojson = {'type': 'Feature', 'geometry': {'type': 'MultiLineString', 'coordinates': ends[ids].tolist()}, 'properties': {}}
    folium.GeoJson(geojson, style_function=lambda _: {'color': '#aeb8bb', 'weight': 1, 'opacity': .45}).add_to(roads)
    roads.add_to(m)
    if reference is not None:
        ref = folium.FeatureGroup(name='Supplied reference route (development only)', show=True)
        geometry = {'type': 'Feature', 'geometry': {'type': 'MultiLineString', 'coordinates': nodes[arcs[reference]].tolist()}, 'properties': {}}
        folium.GeoJson(geometry, style_function=lambda _: {'color': '#b45b8d', 'weight': 6, 'opacity': .45}).add_to(ref)
        ref.add_to(m)
    for title, route, color, show in [('Classifier only', p.classifier_route, '#da9a3b', False), ('HMM / Viterbi', p.decoded_route, '#09877e', True)]:
        group = folium.FeatureGroup(name=title, show=show)
        for piece in route['pieces']:
            if len(piece) > 1:
                folium.PolyLine([[row[1], row[0]] for row in piece], color=color, weight=4, opacity=.9).add_to(group)
        group.add_to(m)
    folium.Marker(latlon[0].tolist(), tooltip='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(latlon[-1].tolist(), tooltip='End', icon=folium.Icon(color='red', icon='stop')).add_to(m)
    legend = f'''<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:16px 20px;border:1px solid #ccd5d4;border-radius:8px;font:13px/1.6 Arial;box-shadow:0 4px 18px #0001;max-width:300px"><b>PathSnap / Trip {html.escape(p.trip.trip_id)}</b><br><span style="color:#467cb5">Blue</span> Raw GPS &nbsp; <span style="color:#aeb8bb">Gray</span> Roads<br><span style="color:#09877e">Teal</span> HMM &nbsp; <span style="color:#da9a3b">Amber</span> Classifier<br><span style="color:#b45b8d">Purple</span> Supplied development reference<br><small>Route breaks are preserved. References are not exposed by the evaluator API.</small></div>'''
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[latlon[:, 0].min(), latlon[:, 1].min()], [latlon[:, 0].max(), latlon[:, 1].max()]], padding=(65, 65))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(destination))
    return destination

def extract_arc_geometries(trip):
    """Return arc_id -> [(lon, lat), ...] for road-map rendering."""
    return {
        arc_id: trip.nodes[trip.arcs[arc_id]].tolist()
        for arc_id in range(len(trip.arcs))
    }
