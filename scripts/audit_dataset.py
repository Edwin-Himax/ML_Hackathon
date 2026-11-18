from pathlib import Path
import numpy as np
import json,csv,hashlib
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Student folder'/'Student folder'
OUT=ROOT/'results'/'dataset_audit'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]; issues=[]; intervals=Counter(); hashes={}; coord_bounds=[]; examples=[]
for partition in ['train','student_test']:
 for folder in sorted((DATA/partition).iterdir()):
  if not folder.is_dir(): continue
  tid=folder.name
  try:
   files={e:folder/(tid+'.'+e) for e in ['track','nodes','arcs','route']}
   missing=[e for e,p in files.items() if not p.is_file()]
   if missing: issues.append({'partition':partition,'trip':tid,'missing':missing}); continue
   tr=np.loadtxt(files['track'],ndmin=2); nd=np.loadtxt(files['nodes'],ndmin=2)
   ar_raw=np.loadtxt(files['arcs'],ndmin=2); rt_raw=np.loadtxt(files['route'],ndmin=1)
   if tr.shape[1]!=3 or nd.shape[1]!=2 or ar_raw.shape[1]!=2 or rt_raw.ndim!=1: raise ValueError('unexpected column count')
   if not np.isfinite(ar_raw).all() or not np.isfinite(rt_raw).all() or not (ar_raw==np.floor(ar_raw)).all() or not (rt_raw==np.floor(rt_raw)).all(): raise ValueError('invalid integer identifiers')
   ar=ar_raw.astype(np.int64); rt=rt_raw.astype(np.int64)
   bad_ar=int(((ar<0)|(ar>=len(nd))).any(axis=1).sum()); bad_rt=int(((rt<0)|(rt>=len(ar))).sum())
   if bad_ar or bad_rt: raise ValueError(f'out-of-range IDs: arcs={bad_ar},route={bad_rt}')
   dt=np.diff(tr[:,2]); intervals.update(float(x) for x in dt)
   lat0=np.radians(np.mean(nd[:,1])); scale=np.array([111195*np.cos(lat0),111195])
   xy=(nd-nd[0])*scale; gps=(tr[:,:2]-nd[0])*scale
   step=np.linalg.norm(np.diff(gps,axis=0),axis=1)
   speed=np.divide(step,dt,out=np.full(len(dt),np.nan),where=dt>0)
   graph=coo_matrix((np.ones(len(ar)),(ar[:,0],ar[:,1])),shape=(len(nd),len(nd))).tocsr()
   ncomp,components=connected_components(graph,directed=True,connection='weak')
   r_ar=ar[rt]; discontinuity=(r_ar[:-1,1]!=r_ar[1:,0])
   revset=set(map(tuple,ar.tolist())); reciprocal=sum((int(v),int(u)) in revset for u,v in ar)
   a=xy[r_ar[:,0]]; b=xy[r_ar[:,1]]; vec=b-a; sq=(vec*vec).sum(axis=1)
   # Exact minimum distance to every supplied route segment for a bounded deterministic observation sample.
   idx=np.unique(np.linspace(0,len(gps)-1,min(200,len(gps))).astype(int)); distances=[]
   for inds in np.array_split(idx,max(1,(len(idx)+19)//20)):
    diff=gps[inds,None,:]-a[None,:,:]
    frac=np.divide((diff*vec).sum(axis=2),sq,out=np.zeros(diff.shape[:2]),where=sq>0).clip(0,1)
    dist=np.sqrt(((diff-frac[:,:,None]*vec)**2).sum(axis=2)).min(axis=1)
    distances.extend(dist.tolist())
   h=hashlib.sha256(files['nodes'].read_bytes()+b'\0'+files['arcs'].read_bytes()).hexdigest()
   hashes.setdefault(h,[]).append(partition+'/'+tid)
   coord_bounds.append([float(nd[:,0].min()),float(nd[:,1].min()),float(nd[:,0].max()),float(nd[:,1].max())])
   row=dict(partition=partition,trip_id=tid,gps_points=len(tr),nodes=len(nd),arcs=len(ar),route_arcs=len(rt),route_equals_track_length=len(rt)==len(tr),nonfinite_track=int((~np.isfinite(tr)).any(axis=1).sum()),nonfinite_nodes=int((~np.isfinite(nd)).any(axis=1).sum()),nonpositive_time_steps=int((dt<=0).sum()),time_gaps_gt1=int((dt>1).sum()),max_time_gap_s=float(dt.max()),duration_s=float(tr[-1,2]-tr[0,2]),speed_gt60_mps=int((speed>60).sum()),max_speed_mps=float(np.nanmax(speed)),weak_components=ncomp,isolated_nodes=int(np.bincount(ar.ravel(),minlength=len(nd)).__eq__(0).sum()),largest_component_nodes=int(np.bincount(components).max()),reciprocal_arc_fraction=reciprocal/len(ar),route_adjacent_disconnects=int(discontinuity.sum()),route_repeated_arc_occurrences=len(rt)-len(set(rt.tolist())),zero_length_arcs=int((np.linalg.norm(xy[ar[:,1]]-xy[ar[:,0]],axis=1)==0).sum()),sample_route_dist_median_m=float(np.median(distances)),sample_route_dist_p95_m=float(np.percentile(distances,95)),sample_route_dist_max_m=float(np.max(distances)),network_hash=h)
   rows.append(row)
   if len(examples)<3: examples.append({'partition':partition,'trip_id':tid,'track_head':tr[:3].tolist(),'route_head':rt[:10].tolist()})
  except Exception as e: issues.append({'partition':partition,'trip':tid,'error':str(e)})
with (OUT/'trip_inventory.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary={'partitions':{},'issues':issues,'unique_networks':len(hashes),'identical_network_groups':[v for v in hashes.values() if len(v)>1],'trip_id_overlap':sorted(set(r['trip_id'] for r in rows if r['partition']=='train')&set(r['trip_id'] for r in rows if r['partition']=='student_test')),'time_step_counts':dict(sorted(intervals.items())),'network_bbox_lon_lat':[min(b[0] for b in coord_bounds),min(b[1] for b in coord_bounds),max(b[2] for b in coord_bounds),max(b[3] for b in coord_bounds)],'examples':examples}
for part in ['train','student_test']:
 rr=[r for r in rows if r['partition']==part]
 agg={'trips':len(rr),'gps_points':sum(r['gps_points'] for r in rr),'route_arcs':sum(r['route_arcs'] for r in rr),'track_route_equal_length_trips':sum(r['route_equals_track_length'] for r in rr)}
 for k in ['gps_points','nodes','arcs','route_arcs','max_time_gap_s','max_speed_mps','weak_components','reciprocal_arc_fraction','sample_route_dist_p95_m']:
  vals=[r[k] for r in rr];agg[k+'_range']=[min(vals),float(np.median(vals)),max(vals)]
 for k in ['nonfinite_track','nonfinite_nodes','nonpositive_time_steps','time_gaps_gt1','speed_gt60_mps','isolated_nodes','route_adjacent_disconnects','route_repeated_arc_occurrences','zero_length_arcs']: agg[k]=sum(r[k] for r in rr)
 summary['partitions'][part]=agg
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['identical_network_groups','examples','time_step_counts']},indent=2))
print('Most common time steps:',intervals.most_common(8))
