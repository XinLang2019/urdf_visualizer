/* Global minimum-jerk splines with joint bounds, IK contact collocation and retiming.
 * This is kinematic optimization, not a floating-base dynamics/contact-force solve. */
window.createTrajectoryOptimizer=function(T,core){
  'use strict';
  const phases=['free','support','flight'],clone=core.clone;
  const signature=p=>JSON.stringify([p.rotation,p.reference||null,p.model.descriptionHash||null,p.model.joints,p.keyframes.map(f=>[f.time,f.base,f.joints,f.interpolation,f.phase||'free',f.supports||null])]);
  const values=f=>[...f.base.position,...f.base.rpy,...f.joints];
  const coefficients=(p,q,v,w,a,b,h)=>{const d=q-p;return [p,h*v,h*h*a/2,10*d-h*(6*v+4*w)-h*h*(1.5*a-.5*b),-15*d+h*(8*v+7*w)+h*h*(1.5*a-b),6*d-h*(3*v+3*w)-h*h*(.5*a-.5*b)];};
  const polynomial=(c,u,d=0)=>{let s=0;for(let i=c.length-1;i>=d;i--){let k=1;for(let j=0;j<d;j++)k*=i-j;s=s*u+c[i]*k;}return s;};
  const bezier=(p,q,v,w,a,b,h)=>[p,p+h*v/5,p+2*h*v/5+h*h*a/20,q-2*h*w/5+h*h*b/20,q-h*w/5,q];
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const gram=(a,b)=>a.reduce((s,v,i)=>s+b.reduce((t,w,j)=>t+v*w/(i+j+1),0),0);
  const jerk=c=>[6*c[3],24*c[4],60*c[5]];
  const defer=()=>new Promise(resolve=>setTimeout(resolve,0));
  function factor(matrix){
    const n=matrix.length,L=matrix.map(()=>new Float64Array(n));
    for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let x=matrix[i][j];for(let k=0;k<j;k++)x-=L[i][k]*L[j][k];if(i===j){if(!(x>0))throw Error('Ill-conditioned timing; use more even keyframe intervals');L[i][j]=Math.sqrt(x);}else L[i][j]=x/L[j][j];}
    return rhs=>{const x=Float64Array.from(rhs);for(let i=0;i<n;i++){for(let j=0;j<i;j++)x[i]-=L[i][j]*x[j];x[i]/=L[i][i];}for(let i=n-1;i>=0;i--){for(let j=i+1;j<n;j++)x[i]-=L[j][i]*x[j];x[i]/=L[i][i];}return x;};
  }
  function system(times){
    const n=times.length,size=2*(n-2),intervals=times.slice(1).map((t,i)=>t-times[i]),unit=intervals.slice().sort((a,b)=>a-b)[Math.floor(intervals.length/2)];
    const H=Array.from({length:size},()=>new Float64Array(size)),segments=[];
    const index=(i,d)=>i===0||i===n-1?-1:2*(i-1)+d;
    for(let k=0;k<n-1;k++){
      const h=intervals[k]/unit,ids=[index(k,0),index(k,1),index(k+1,0),index(k+1,1)];
      const basis=[jerk(coefficients(0,0,1,0,0,0,h)),jerk(coefficients(0,0,0,0,1,0,h)),jerk(coefficients(0,0,0,1,0,0,h)),jerk(coefficients(0,0,0,0,0,1,h))],weight=1/h**5;
      for(let i=0;i<4;i++)if(ids[i]>=0)for(let j=0;j<4;j++)if(ids[j]>=0)H[ids[i]][ids[j]]+=gram(basis[i],basis[j])*weight;
      segments.push({h,ids,basis,weight,k});
    }
    const solve=factor(H),rows=[];
    for(const s of segments){const {h,k}=s;for(const [i,a,b,endpoint]of [[k,h/5,0,k],[k,2*h/5,h*h/20,k],[k+1,-2*h/5,h*h/20,k+1],[k+1,-h/5,0,k+1]]){
      const row=new Float64Array(size),iv=index(i,0),ia=index(i,1);if(iv>=0)row[iv]=a;if(ia>=0)row[ia]=b;
      if(!row.some(v=>v))continue;const direction=solve(row),denom=dot(row,direction);rows.push({row,direction,denom,endpoint});
    }}
    return {n,size,unit,segments,solve,rows};
  }
  function channel(sys,positions,lower=null,upper=null){
    const g=new Float64Array(sys.size);
    for(const s of sys.segments){const base=jerk(coefficients(positions[s.k],positions[s.k+1],0,0,0,0,s.h));s.ids.forEach((id,i)=>{if(id>=0)g[id]-=gram(s.basis[i],base)*s.weight;});}
    const x=sys.solve(g),constraints=[];
    for(const row of sys.rows){if(Number.isFinite(upper))constraints.push({...row,sign:1,bound:upper-positions[row.endpoint],dual:0});if(Number.isFinite(lower))constraints.push({...row,sign:-1,bound:positions[row.endpoint]-lower,dual:0});}
    // Hildreth dual coordinate descent: the metric is the jerk Hessian, not
    // an arbitrary blend of interpolation slopes. x=0 is position-feasible.
    let converged=!constraints.length;
    for(let sweep=0;!converged&&sweep<1500;sweep++){
      let deltaMax=0;
      for(const c of constraints){const next=Math.max(0,c.dual+(c.sign*dot(c.row,x)-c.bound)/c.denom),delta=next-c.dual;c.dual=next;deltaMax=Math.max(deltaMax,Math.abs(delta)*Math.sqrt(c.denom));if(delta)for(let i=0;i<x.length;i++)x[i]-=delta*c.sign*c.direction[i];}
      converged=deltaMax<1e-8&&constraints.every(c=>c.sign*dot(c.row,x)-c.bound<1e-8);
    }
    if(!converged)throw Error('Joint-bound optimization did not converge; adjust the poses or key times');
    return positions.map((p,i)=>[p,i===0||i===sys.n-1?0:x[2*(i-1)]/sys.unit,i===0||i===sys.n-1?0:x[2*(i-1)+1]/sys.unit**2]);
  }
  async function spline(times,points,joints,cancelled){
    const sys=system(times),knots=times.map((time,i)=>({time,position:points[i].slice(),velocity:[],acceleration:[]}));
    for(let d=0;d<points[0].length;d++){
      const joint=joints[d-6],out=channel(sys,points.map(p=>p[d]),joint?.lower,joint?.upper);
      out.forEach((v,i)=>{knots[i].velocity[d]=v[1];knots[i].acceleration[d]=v[2];});
      if(d%4===3){await defer();if(cancelled())throw Error('Optimization cancelled');}
    }
    return knots;
  }
  function evaluate(knots,time){
    let i=0;while(i<knots.length-2&&knots[i+1].time<=time)i++;
    const a=knots[i],b=knots[i+1],h=b.time-a.time,u=Math.max(0,Math.min(1,(time-a.time)/h));
    const out={position:[],velocity:[],acceleration:[]};
    a.position.forEach((v,d)=>{const c=coefficients(v,b.position[d],a.velocity[d],b.velocity[d],a.acceleration[d],b.acceleration[d],h);out.position[d]=polynomial(c,u);out.velocity[d]=time<0||time>knots.at(-1).time?0:polynomial(c,u,1)/h;out.acceleration[d]=time<0||time>knots.at(-1).time?0:polynomial(c,u,2)/h**2;});
    return out;
  }
  function bounds(knots,model,settings){
    const peakV=knots[0].position.map(()=>0),peakA=peakV.slice();let scale=1,cost=0;
    for(let k=0;k<knots.length-1;k++){
      const a=knots[k],b=knots[k+1],h=b.time-a.time;
      a.position.forEach((p,d)=>{
        const controls=bezier(p,b.position[d],a.velocity[d],b.velocity[d],a.acceleration[d],b.acceleration[d],h),joint=model.joints[d-6];
        if(joint&&controls.some(v=>Number.isFinite(joint.lower)&&v<joint.lower-2e-7||Number.isFinite(joint.upper)&&v>joint.upper+2e-7))throw Error('Position bound failed: '+joint.name);
        const v=controls.slice(1).map((p,i)=>5*(p-controls[i])/h),acc=v.slice(1).map((p,i)=>4*(p-v[i])/h);
        peakV[d]=Math.max(peakV[d],...v.map(Math.abs));peakA[d]=Math.max(peakA[d],...acc.map(Math.abs));
        const maxV=d<3?settings.baseSpeed:d<6?settings.angularSpeed:Math.min(settings.jointSpeed,Number.isFinite(joint.velocity)&&joint.velocity>0?joint.velocity:Infinity),maxA=d<3?settings.baseAcceleration:d<6?settings.angularAcceleration:settings.jointAcceleration;
        scale=Math.max(scale,peakV[d]/maxV,Math.sqrt(peakA[d]/maxA));
        const c=jerk(coefficients(p,b.position[d],a.velocity[d],b.velocity[d],a.acceleration[d],b.acceleration[d],h));cost+=gram(c,c)/h**5;
      });
    }
    const baseSpeedBound=Math.hypot(...peakV.slice(0,3)),baseAccelerationBound=Math.hypot(...peakA.slice(0,3));
    // RPY rates are not spatial angular velocity. These triangle bounds also
    // include the moving-axis cross terms in spatial angular acceleration.
    const angularSpeedBound=peakV[3]+peakV[4]+peakV[5],angularAccelerationBound=peakA[3]+peakA[4]+peakA[5]+(peakV[5]+peakV[4])*peakV[3]+peakV[5]*peakV[4];
    scale=Math.max(scale,baseSpeedBound/settings.baseSpeed,Math.sqrt(baseAccelerationBound/settings.baseAcceleration),angularSpeedBound/settings.angularSpeed,Math.sqrt(angularAccelerationBound/settings.angularAcceleration));
    return {scale,peakV,peakA,cost,baseSpeedBound,baseAccelerationBound,angularSpeedBound,angularAccelerationBound};
  }
  function settings(input={}){
    const s={jointSpeed:3,jointAcceleration:12,baseSpeed:1.5,baseAcceleration:6,angularSpeed:6,angularAcceleration:20,autoRetime:true,...input};
    for(const name of ['jointSpeed','jointAcceleration','baseSpeed','baseAcceleration','angularSpeed','angularAcceleration'])if(!Number.isFinite(s[name])||s[name]<=0)throw Error('Optimizer limits must be finite and positive');
    if(typeof s.autoRetime!=='boolean')throw Error('Automatic retiming must be boolean');
    return s;
  }
  function validate(p){
    const o=p.optimization;if(!o)return;
    const n=6+p.model.joints.length;
    if(o.schema!=='urdf-viewer.optimization.v1'||o.signature!==signature(p)||p.rotation!=='rpy'||!Array.isArray(o.knots)||o.knots.length<2||o.knots.length>160)throw Error('Invalid or stale optimized trajectory');
    const config=settings(o.settings),r=o.report;
    if(!r||!Number.isFinite(r.originalDuration)||r.originalDuration<=0||!Number.isFinite(r.duration)||Math.abs(r.duration-core.duration(p))>1e-8||!Number.isFinite(r.timeScale)||r.timeScale<1||Math.abs(r.originalDuration*r.timeScale-r.duration)>1e-7||!Number.isFinite(r.contactError)||r.contactError<0||r.contactError>.0020001||r.minimumGroundClearance!==null&&(!Number.isFinite(r.minimumGroundClearance)||r.minimumGroundClearance<-.0020001)||!Number.isInteger(r.contactConstraints)||r.contactConstraints<0)throw Error('Invalid optimization report');
    let last=-1;
    for(const k of o.knots){if(!Number.isFinite(k.time)||k.time<=last||['position','velocity','acceleration'].some(field=>!Array.isArray(k[field])||k[field].length!==n||!k[field].every(Number.isFinite)))throw Error('Invalid optimized knots');last=k.time;}
    if(o.knots[0].time!==0||Math.abs(last-core.duration(p))>1e-8)throw Error('Optimized timeline mismatch');
    const b=bounds(o.knots,p.model,config);if(!Number.isFinite(b.scale)||b.scale>1.00001)throw Error('Optimized trajectory exceeds speed or acceleration limits');
    for(const f of p.keyframes){const value=evaluate(o.knots,f.time).position;if(values(f).slice(0,6).some((x,i)=>Math.abs(x-value[i])>1e-6))throw Error('Optimized base pose does not match source keys');}
    r.kinematicOnly=true;r.dynamicsValidated=false;r.selfCollisionValidated=false;
  }
  async function optimize(project,input={},kinematics=null,progress=()=>{},cancelled=()=>false){
    const p=clone(project);delete p.optimization;core.validate(p,p.model);
    const config=settings(input),fs=p.keyframes;if(fs.length<2||fs.length>60)throw Error('Optimization requires 2–60 keyframes');
    if(p.model.unsupported?.length)throw Error('Internal floating/planar joints are not supported by the optimizer');
    if(p.rotation!=='rpy')throw Error('Select unwrapped RPY before optimization (full turns are retained)');
    if(fs.some((f,i)=>i&&f.time-fs[i-1].time<.01))throw Error('Optimizer keyframe intervals must be at least 0.01 s');
    if(fs.some(f=>f.phase==='support')&&!p.reference)throw Error('Set a fixed ground before optimizing support phases');
    if(!kinematics&&fs.some(f=>f.phase==='support'))throw Error('Contact optimization requires robot kinematics');
    progress('smoothing');let knots=await spline(fs.map(f=>f.time),fs.map(values),p.model.joints,cancelled);
    const subdivisions=Math.max(1,Math.min(8,Math.floor(100/(fs.length-1)))),times=[],points=[],targets=[];
    let contactCount=0,maxIKChange=0;
    if(kinematics){
      progress('contacts');
      for(let i=0;i<fs.length-1;i++){
        const f=fs[i],names=f.phase==='support'?(f.supports||p.reference.supports||[]):[];
        if(f.phase==='support'&&!names.length)throw Error('A support phase needs at least one selected foot');
        if(names.length){kinematics.set(values(f));for(const name of names){const point=kinematics.point(name),previous=targets.find(t=>t.name===name&&Math.abs(t.end-f.time)<1e-9);targets.push({start:f.time,end:fs[i+1].time,name,position:previous?previous.position.slice():[point[0],point[1],0]});}contactCount+=names.length;}
        for(let j=0;j<subdivisions;j++){const t=f.time+(fs[i+1].time-f.time)*j/subdivisions;times.push(t);points.push(evaluate(knots,t).position);}
      }
      times.push(fs.at(-1).time);points.push(values(fs.at(-1)));
      // Both sides of a phase boundary constrain their common knot. Incompatible
      // contact anchors are reported, never silently reset at the next segment.
      for(let i=0;i<points.length;i++){
        const active=targets.filter(t=>times[i]>=t.start-1e-9&&times[i]<=t.end+1e-9);
        if(active.length){const before=points[i].slice();points[i]=kinematics.project(points[i],active);maxIKChange=Math.max(maxIKChange,...points[i].slice(6).map((v,j)=>Math.abs(v-before[j+6])));}
        if(i%4===0){await defer();if(cancelled())throw Error('Optimization cancelled');}
      }
      // No-contact motion keeps the smaller global spline instead of creating
      // artificial stops or redundant knots.
      if(targets.length)knots=await spline(times,points,p.model.joints,cancelled);
    }
    const limits=bounds(knots,p.model,config),scale=limits.scale>1?limits.scale*1.001:1;
    if(scale>1.00001&&!config.autoRetime)throw Error('Timing violates limits; enable automatic retiming (required scale '+scale.toFixed(2)+'×)');
    if(core.duration(p)*scale>3600)throw Error('Retimed motion exceeds 3600 seconds');
    progress('checking');let contactError=0,minClearance=Infinity,checks=0;
    if(kinematics){
      for(let k=0;k<knots.length-1;k++)for(let j=0;j<=6;j++){
        const t=knots[k].time+(knots[k+1].time-knots[k].time)*j/6,v=evaluate(knots,t).position;kinematics.set(v);
        for(const target of targets.filter(a=>t>=a.start-1e-9&&t<=a.end+1e-9)){const point=kinematics.point(target.name);contactError=Math.max(contactError,Math.hypot(...point.map((x,d)=>x-target.position[d])));}
        if(p.reference)minClearance=Math.min(minClearance,kinematics.clearance());checks++;
        if(checks%16===0){await defer();if(cancelled())throw Error('Optimization cancelled');}
      }
      if(contactError>.002)throw Error('Foot-lock residual '+(contactError*1000).toFixed(2)+' mm exceeds 2 mm; add intermediate poses or reduce base motion');
      if(minClearance<-.002)throw Error('Ground penetration '+(-minClearance*1000).toFixed(2)+' mm; raise the base or change the pose');
    }
    fs.forEach(f=>f.time*=scale);knots.forEach(k=>{k.time*=scale;k.velocity=k.velocity.map(v=>v/scale);k.acceleration=k.acceleration.map(v=>v/scale**2);});
    const report={method:'global minimum integrated squared jerk; joint position constraints via Bezier convex hull; contact IK collocation',timeScale:scale,originalDuration:core.duration(project),duration:core.duration(p),contactError,minimumGroundClearance:Number.isFinite(minClearance)?minClearance:null,contactConstraints:contactCount,collisionSamples:checks,maximumJointCorrection:maxIKChange,peakVelocityBounds:limits.peakV.map(v=>v/scale),peakAccelerationBounds:limits.peakA.map(v=>v/scale**2),baseSpeedBound:limits.baseSpeedBound/scale,baseAccelerationBound:limits.baseAccelerationBound/scale**2,angularSpeedBound:limits.angularSpeedBound/scale,angularAccelerationBound:limits.angularAccelerationBound/scale**2,jerkCost:limits.cost/scale**5,kinematicOnly:true,dynamicsValidated:false,selfCollisionValidated:false,contactValidation:'sampled at 6 subintervals per polynomial segment; not a continuous-time certificate'};
    p.optimization={schema:'urdf-viewer.optimization.v1',signature:signature(p),settings:config,knots,report};validate(p);return p;
  }
  return {optimize,evaluate,validate,signature,values,settings,bounds,phases};
};
