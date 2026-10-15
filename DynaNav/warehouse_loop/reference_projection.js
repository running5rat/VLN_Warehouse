/* Source-body FLU XY -> optical camera -> original RGB pixels. Display only. */
(function (root) {
  'use strict';
  function resample(points) {
    const p = [];
    for (const q of points) {
      if (!Array.isArray(q) || q.length !== 2 || !q.every(Number.isFinite)) throw Error('参考线坐标必须为有限数值');
      if (!p.length || Math.hypot(q[0]-p.at(-1)[0], q[1]-p.at(-1)[1]) > 1e-5) p.push(q);
    }
    if (p.length < 2) return [];
    const s = [0];
    for (let i=1;i<p.length;i++) s.push(s.at(-1)+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]));
    const end = Math.min(s.at(-1), 7.75), distances = [];
    for (let d=0;d<end;d+=.25) distances.push(d);
    distances.push(end);
    return distances.slice(0,32).map(d=>interpolate(p,s,d));
  }
  function interpolate(p,s,d) {
    let j=1;while(j<s.length-1 && s[j]<d) j++;
    const t=(d-s[j-1])/(s[j]-s[j-1] || 1);
    return p[j].map((v,k)=>p[j-1][k]*(1-t)+v*t);
  }
  function anchors(p) {
    if(p.length<2)return [];
    const s=[0];for(let i=1;i<p.length;i++)s.push(s.at(-1)+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]));
    if(s.at(-1)<1e-5)return [];
    return [1,2,3,4,5].map(k=>interpolate(p,s,s.at(-1)*k/5).map(v=>+v.toFixed(3)));
  }
  function project(points, camera) {
    const T=camera.T_body_camera,K=camera.K,w=camera.width,h=camera.height;
    if(!T || T.length!==4 || !K || K.length!==3 || !(w>0&&h>0))throw Error('投影标定缺失');
    return points.map(p=>{
      // Reference is already expressed in the request's source-body frame.
      // Lift XY to z=0; inverse rigid transform is R^T (p - translation).
      const body=[p[0],p[1],0], q=[0,0,0];
      for(let j=0;j<3;j++)for(let i=0;i<3;i++)q[j]+=T[i][j]*(body[i]-T[i][3]);
      if(!q.every(Number.isFinite))throw Error('投影得到非有限坐标');
      if(q[2]<=.05)return {uv:null,status:'behind_or_near',depth:q[2]};
      const z=K[2][0]*q[0]+K[2][1]*q[1]+K[2][2]*q[2];
      const uv=[0,1].map(i=>(K[i][0]*q[0]+K[i][1]*q[1]+K[i][2]*q[2])/z);
      return {uv,status:uv[0]>=0&&uv[0]<w&&uv[1]>=0&&uv[1]<h?'inside':'outside',depth:q[2]};
    });
  }
  root.ReferenceProjection={resample,anchors,project};
})(globalThis);
