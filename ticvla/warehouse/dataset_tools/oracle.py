"""Lattice seeds, official ALTRO file bridge, and independent candidate checks.

Solver failures are unknown, not proofs that a mode is infeasible. Privileged
future obstacles are accepted only by the separate evaluation function.
"""

import math
from pathlib import Path
import subprocess
import time

import numpy as np

from ..schema import file_sha256, fingerprint
from .core import read_json, write_json

LIMITS = dict(v_max=.3, w_max=.6, a_max=.8, alpha_max=1.5, radius=.35)


def predict_tracks(tracks, t_obs, horizon=30, dt=.1):
    predicted = []
    for tr in tracks:
        if tr.get("source") != "rgbd_tracking":
            raise ValueError("Planner tracks must originate from RGB-D observations")
        stamps, positions = np.asarray(tr["times"], float), np.asarray(tr["positions"], float)
        if len(stamps) < 2 or positions.shape != (len(stamps), 2):
            raise ValueError("At least two synchronized target observations required")
        if not np.isfinite(stamps).all() or not np.isfinite(positions).all() or np.any(np.diff(stamps)<=0):
            raise ValueError("Invalid dynamic observation sequence")
        if stamps[-1] > t_obs or t_obs-stamps[-1] > .5:
            raise ValueError("Future/stale dynamic observation")
        if not np.isfinite([tr["radius"], tr.get("position_uncertainty_m", .1)]).all() or tr["radius"] <= 0 or tr.get("position_uncertainty_m", .1) < 0:
            raise ValueError("Invalid obstacle radius or uncertainty")
        v = (positions[-1]-positions[-2])/(stamps[-1]-stamps[-2])
        times = t_obs-stamps[-1]+np.arange(horizon+1)*dt
        xy = positions[-1]+times[:,None]*v
        uncertainty = float(tr.get("position_uncertainty_m", .1)) + times*.1
        predicted.append(dict(xy=xy.tolist(), radius=(float(tr["radius"])+uncertainty).tolist()))
    return predicted


def relative_offsets(xy, reference):
    p, ref = np.asarray(xy), np.asarray(reference)
    if len(ref)<2: raise ValueError("Nominal reference required; do not infer it from a dynamic detour")
    a, d = ref[:-1], np.diff(ref, axis=0)
    lens = np.linalg.norm(d, axis=1)
    if not np.any(lens>1e-5): raise ValueError("Degenerate reference")
    fractions = np.clip(((p[:,None]-a)*d).sum(-1)/np.maximum(lens**2,1e-10),0,1)
    projections = a + fractions[...,None]*d
    distances = np.linalg.norm(p[:,None]-projections,axis=-1)
    chosen = distances.argmin(1)
    tangent = d[chosen]/np.maximum(lens[chosen,None],1e-10)
    delta = p-projections[np.arange(len(p)),chosen]
    lateral = tangent[:,0]*delta[:,1]-tangent[:,1]*delta[:,0]
    progress = np.r_[0,np.cumsum(lens)][chosen]+fractions[np.arange(len(p)),chosen]*lens[chosen]
    return lateral, progress


def classify_mode(states, reference, *, dt=.1):
    x = np.asarray(states, float)
    lateral, progress = relative_offsets(x[:, :2], reference)
    peak = float(lateral[np.argmax(np.abs(lateral))])
    lat = 0 if peak > .15 else 2 if peak < -.15 else 1
    speed = np.mean(np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1))/dt
    if abs(x[-1,3]) < .03 and (speed < .03 or x[0,3] > .05): lon = 0
    elif speed >= .225: lon = 2
    elif .05 <= speed <= .20: lon = 1
    else: return None
    return lat*3+lon


def lattice_seeds(reference, state, horizon=30, dt=.1):
    reference = np.asarray(reference, float)
    s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
    if len(s)<2 or s[-1]<1e-4: raise ValueError("Need a nonempty nominal static route")
    seeds=[]
    for lat in range(3):
        for lon in range(3):
            speed = [0., .15, .3][lon]
            dist = min(s[-1], speed*horizon*dt)
            x,y = [np.interp(dist,s,reference[:,i]) for i in (0,1)]
            seg = min(len(reference)-2,max(0,int(np.searchsorted(s,dist)-1)))
            tangent = reference[seg+1]-reference[seg]
            heading = math.atan2(tangent[1],tangent[0])
            offset = [.4,0.,-.4][lat] if lon else 0.
            for scale in (.75,1.,1.25):
                goal = [x-offset*scale*math.sin(heading),y+offset*scale*math.cos(heading),heading,speed,0.]
                # Controls are just seeds, not supervised trajectories.
                controls=[]; v,w=float(state[3]),float(state[4])
                desired_w=np.clip(offset/(max(dist,.3)*horizon*dt),-.5,.5)
                for k in range(horizon):
                    a=float(np.clip((speed-v)/dt,-LIMITS["a_max"],LIMITS["a_max"]))
                    alpha=float(np.clip((desired_w-w)/dt,-LIMITS["alpha_max"],LIMITS["alpha_max"]))
                    controls.append([a,alpha]); v+=a*dt; w+=alpha*dt
                seeds.append(dict(mode_index=lat*3+lon, goal=goal, controls=controls))
    return seeds


def score_candidate(candidate, reference, static_circles, predicted, *, initial_state, dt=.1, privileged=None):
    """Checks actual states/controls independently of optimizer success/status."""
    x,u=np.asarray(candidate.get("states",[]),float),np.asarray(candidate.get("controls",[]),float)
    reasons=[]
    if x.shape!=(31,5) or u.shape!=(30,2) or not np.isfinite(x).all() or not np.isfinite(u).all():
        return dict(status="rejected",reasons=["invalid_shape_or_nonfinite"])
    if not np.allclose(x[0], initial_state, atol=1e-5): reasons.append("initial_state_mismatch")
    if np.any(x[:,3]<-1e-4) or np.max(x[:,3])>LIMITS["v_max"]+1e-4: reasons.append("speed")
    if np.max(np.abs(x[:,4]))>LIMITS["w_max"]+1e-4: reasons.append("angular_speed")
    if np.max(np.abs(u[:,0]))>LIMITS["a_max"]+1e-4: reasons.append("acceleration")
    if np.max(np.abs(u[:,1]))>LIMITS["alpha_max"]+1e-4: reasons.append("angular_acceleration")
    if np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1).sum() < .02 and np.ptp(np.unwrap(x[:,2])) > .1:
        reasons.append("pure_rotation_not_expressible_by_xy_action")
    # Verify integration using an independent RK4 implementation.
    def f(z, control): return np.array([z[3]*np.cos(z[2]),z[3]*np.sin(z[2]),z[4],*control])
    z=x[:-1]; reconstructed=[]
    for a,b in zip(z,u):
        k1=f(a,b); k2=f(a+dt*k1/2,b); k3=f(a+dt*k2/2,b); k4=f(a+dt*k3,b)
        reconstructed.append(a+dt*(k1+2*k2+2*k3+k4)/6)
    residual=float(np.max(np.abs(np.asarray(reconstructed)-x[1:])))
    if residual>1e-3: reasons.append("dynamics_residual")
    times=np.arange(31)*dt
    fine=np.linspace(0,30*dt,301)
    xy=np.column_stack([np.interp(fine,times,x[:,i]) for i in (0,1)])
    clearance=None
    for c in static_circles:
        d=np.linalg.norm(xy-np.asarray(c[:2]),axis=1)-float(c[2])-LIMITS["radius"]
        clearance=float(d.min()) if clearance is None else min(clearance,float(d.min()))
        if d.min() < -1e-4: reasons.append("static_collision")
    for track in predicted:
        p=np.asarray(track["xy"])
        p=np.column_stack([np.interp(fine,times,p[:,i]) for i in (0,1)])
        radii=np.interp(fine,times,track["radius"])
        d=np.linalg.norm(xy-p,axis=1)-radii-LIMITS["radius"]
        clearance=float(d.min()) if clearance is None else min(clearance,float(d.min()))
        if d.min() < -1e-4: reasons.append("predicted_dynamic_collision")
    true_collision=None
    if privileged is not None:
        true_collision=False
        for track in privileged:
            p=np.asarray(track["xy"])
            if p.shape!=(31,2): raise ValueError("Independent future evaluator requires synchronized horizon")
            p=np.column_stack([np.interp(fine,times,p[:,i]) for i in (0,1)])
            if np.any(np.linalg.norm(xy-p,axis=1)<LIMITS["radius"]+float(track["radius"])):
                true_collision=True; reasons.append("future_evaluation_collision")
    lateral,progress=relative_offsets(x[:,:2],reference)
    mode=classify_mode(x,reference)
    if mode is None or mode != candidate.get("mode_index"): reasons.append("unresolved_or_mismatched_mode")
    return dict(status="verified" if not reasons else "rejected", reasons=sorted(set(reasons)),
                actual_mode=mode, dynamics_residual=residual, min_clearance=clearance,
                future_collision=true_collision, progress=float(progress[-1]-progress[0]),
                deviation=float(np.mean(np.abs(lateral))), smoothness=float(np.mean(u**2)),
                dynamics_checked=True)


def run_oracle(request_path, solver_config, output):
    request,config=read_json(request_path),read_json(solver_config)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    required={"source_binding","t_obs","state","reference","static_circles","tracks","observation_frame"}
    if set(request)!=required or request["observation_frame"]!="source_body_flu":
        raise ValueError("Oracle input allowlist violated; future GT belongs only in independent evaluator")
    x0=np.asarray(request["state"],float)
    if x0.shape!=(5,) or not np.isfinite(x0).all() or not np.allclose(x0[:3],0):
        raise ValueError("Expected source-body initial state [0,0,0,v,w]")
    predictions=predict_tracks(request["tracks"],request["t_obs"])
    payload=dict(state=x0.tolist(), reference=request["reference"], static_circles=request["static_circles"],
                 predictions=predictions, dt=.1, horizon=30, limits=LIMITS,
                 seeds=lattice_seeds(request["reference"],x0), source_binding=request["source_binding"])
    write_json(output/"solver-input.json",payload)
    report=dict(status="not_run",source_binding=request["source_binding"],modes=["unknown"]*9,
                request_sha256=file_sha256(request_path),solver_config_sha256=file_sha256(solver_config),
                isaac_execution_verified=False)
    argv=config.get("argv",[])
    if not argv or not Path(argv[0]).is_file():
        report.update(status="blocked",reason="Official ALTRO worker executable unavailable; no fallback planner")
        write_json(output/"report.json",report); return report
    start=time.monotonic()
    try:
        with (output/"solver.log").open("w") as log:
            subprocess.run([*argv,str(output/"solver-input.json"),str(output/"solver-output.json")],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=config.get("timeout_seconds",300))
        raw=read_json(output/"solver-output.json")
        if raw["source_binding"]!=request["source_binding"]: raise ValueError("Solver source mismatch")
        candidates=[]
        for c in raw["candidates"]:
            if c.get("solver_status")!="solved":
                candidates.append(dict(**c,evaluation=dict(status="unknown",reasons=["solver_failed"])))
                continue
            score=score_candidate(c,request["reference"],request["static_circles"],predictions,initial_state=x0)
            candidates.append(dict(**c,evaluation=score))
            if score["status"]=="verified": report["modes"][c["mode_index"]]="positive"
        good=[(i,c) for i,c in enumerate(candidates) if c["evaluation"]["status"]=="verified"]
        ranking=sorted(good,key=lambda pair:(-pair[1]["evaluation"]["progress"],pair[1]["evaluation"]["deviation"],pair[1]["evaluation"]["smoothness"],pair[0]))
        write_json(output/"candidates.json",candidates)
        report.update(status="offline_checked",verified_candidates=len(good),ranking=[i for i,c in ranking],
                      future_truth_used_by_planner=False,quality_gt_ready=False,
                      reason="Requires independent dynamic truth evaluation and Isaac execution before action GT approval")
    except (subprocess.SubprocessError,ValueError,KeyError,OSError) as exc:
        report.update(status="failed",reason=str(exc))
    report["elapsed_seconds"]=time.monotonic()-start
    write_json(output/"report.json",report)
    return report
