/* Deterministic, DOM-free reference trajectory evaluation. All angles are radians. */
window.createTrajectoryCore=function(T){
  'use strict';
  const modes=['linear','smoothstep','quintic','cubic'],rotations=['rpy','slerp'];
  const finite=Number.isFinite,clone=x=>JSON.parse(JSON.stringify(x));
  const quaternion=rpy=>new T.Quaternion().setFromEuler(new T.Euler(...rpy,'ZYX'));
  const duration=p=>p?.plan?.frames.at(-1)?.time||p?.keyframes.at(-1)?.time||0;
  let planMath;const planner=()=>planMath||(planMath=createWholeBodyTrajectoryCore(T));
  let optimizationMath;
  const optimizer=()=>optimizationMath||(optimizationMath=createTrajectoryOptimizer(T,{clone,duration}));
  function validate(p,model){
    if(!p||p.schema!=='urdf-viewer.motion.v1')throw Error('Invalid motion project schema');
    if(typeof p.name!=='string'||p.name.length>160||!Array.isArray(p.keyframes)||p.keyframes.length>2000)throw Error('Invalid project or keyframe count');
    if(!rotations.includes(p.rotation)||!finite(p.fps)||p.fps<1||p.fps>1000)throw Error('Invalid orientation mode or sample rate');
    if(p.reference!==undefined&&p.reference!==null){
      const r=p.reference;
      if(r.kind!=='fixed-ground'||!Array.isArray(r.origin)||r.origin.length!==3||!r.origin.every(finite)||!Array.isArray(r.quaternion)||r.quaternion.length!==4||!r.quaternion.every(finite)||Math.abs(Math.hypot(...r.quaternion)-1)>1e-6)throw Error('Invalid fixed-ground reference transform');
    }
    if(!p.model||p.model.name!==model.name||p.model.root!==model.root||!Array.isArray(p.model.joints)||p.model.joints.length!==model.joints.length||p.model.joints.some((j,i)=>j.name!==model.joints[i].name||j.type!==model.joints[i].type))throw Error('Model / joint order mismatch');
    let previous=-1;
    const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
    const validPose=f=>{
      if(!vector(f.base?.position,3)||!vector(f.base?.rpy,3)||!vector(f.joints,model.joints.length))throw Error('Invalid pose values');
      f.joints.forEach((v,i)=>{const j=model.joints[i];if((finite(j.lower)&&v<j.lower-1e-9)||(finite(j.upper)&&v>j.upper+1e-9))throw Error('Joint outside URDF limits: '+j.name);});
    };
    if(p.reference?.initialStandingPose)validPose(p.reference.initialStandingPose);
    for(const f of p.keyframes){
      if(!finite(f.time)||f.time<0||f.time>3600||f.time-previous<1e-4||!modes.includes(f.interpolation)||typeof f.label!=='string'||f.label.length>160)throw Error('Invalid keyframe time, order or interpolation');
      if(f.phase!==undefined&&!['free','support','flight'].includes(f.phase)||f.supports!==undefined&&(!Array.isArray(f.supports)||f.supports.some(n=>typeof n!=='string'||!p.reference?.supports?.includes(n))))throw Error('Invalid support phase or foot selection');
      validPose(f);
      previous=f.time;
    }
    if(p.keyframes.length&&p.keyframes[0].time!==0)throw Error('The first keyframe must start at 0 s');
    const result={...clone(p),model:clone(model),loop:!!p.loop};
    if(result.optimizerSettings)optimizer().settings(result.optimizerSettings);
    if(result.optimization)optimizer().validate(result);
    if(result.plan)planner().validate(result.plan,model);
    return result;
  }
  function ease(mode,u){
    if(mode==='smoothstep')return [u*u*(3-2*u),6*u*(1-u),6-12*u];
    if(mode==='quintic')return [u**3*(10+u*(-15+6*u)),30*u*u*(1-u)**2,60*u*(1-u)*(1-2*u)];
    return [u,1,0];
  }
  // Nonuniform, shape-preserving Hermite slopes (PCHIP). No joint overshoot.
  function tangent(frames,k,get){
    const n=frames.length,slope=i=>(get(frames[i+1])-get(frames[i]))/(frames[i+1].time-frames[i].time);
    if(n<2)return 0;if(n===2)return slope(0);
    if(k===0||k===n-1){
      const first=k===0,i=first?0:n-2,j=first?1:n-3,h=frames[i+1].time-frames[i].time,g=frames[j+1].time-frames[j].time,d=slope(i),e=slope(j);
      const m=((2*h+g)*d-h*e)/(h+g);return Math.sign(m)!==Math.sign(d)?0:Math.sign(d)!==Math.sign(e)&&Math.abs(m)>3*Math.abs(d)?3*d:m;
    }
    const a=slope(k-1),b=slope(k),h=frames[k].time-frames[k-1].time,g=frames[k+1].time-frames[k].time;
    if(a*b<=0)return 0;const w1=2*g+h,w2=g+2*h;return (w1+w2)/(w1/a+w2/b);
  }
  function scalar(frames,k,get,u,h,mode){
    const a=get(frames[k]),b=get(frames[k+1]),d=b-a;
    if(mode!=='cubic'){const [s,v,acc]=ease(mode,u);return [a+d*s,d*v/h,d*acc/(h*h)];}
    const m=tangent(frames,k,get)*h,n=tangent(frames,k+1,get)*h,u2=u*u,u3=u2*u;
    return [(2*u3-3*u2+1)*a+(u3-2*u2+u)*m+(-2*u3+3*u2)*b+(u3-u2)*n,
      ((6*u2-6*u)*a+(3*u2-4*u+1)*m+(-6*u2+6*u)*b+(3*u2-2*u)*n)/h,
      ((12*u-6)*a+(6*u-4)*m+(-12*u+6)*b+(6*u-2)*n)/(h*h)];
  }
  function evaluate(p,time){
    if(p.plan){if(!finite(time))throw Error('Invalid sample time');return planner().evaluate(p.plan,time);}
    const fs=p.keyframes;if(!fs.length)throw Error('No keyframes');
    const end=duration(p),t=Math.max(0,Math.min(end,time));
    if(!finite(time))throw Error('Invalid sample time');
    let k=0;while(k<fs.length-2&&fs[k+1].time<=t)k++;
    const stationary=fs.length===1||time<0||time>end;
    const h=stationary?1:fs[k+1].time-fs[k].time,u=stationary?0:(t-fs[k].time)/h,mode=fs[k].interpolation;
    const channel=get=>stationary?[get(time>end?fs.at(-1):fs[0]),0,0]:scalar(fs,k,get,u,h,mode);
    let position,angles,joints;
    if(p.optimization){
      if(p.optimization.signature!==optimizer().signature(p))throw Error('Optimized trajectory is stale; run optimization again');
      const solved=optimizer().evaluate(p.optimization.knots,time),get=i=>[solved.position[i],solved.velocity[i],solved.acceleration[i]];
      position=[0,1,2].map(get);angles=[3,4,5].map(get);joints=p.model.joints.map((j,i)=>get(i+6));
    }else{position=[0,1,2].map(i=>channel(f=>f.base.position[i]));angles=[0,1,2].map(i=>channel(f=>f.base.rpy[i]));joints=p.model.joints.map((j,i)=>channel(f=>f.joints[i]));}
    let rpy=angles.map(c=>c[0]),q=quaternion(rpy),omega,alpha;
    if(p.rotation==='slerp'&&!stationary){
      const a=quaternion(fs[k].base.rpy),b=quaternion(fs[k+1].base.rpy);
      if(a.dot(b)<0)b.set(-b.x,-b.y,-b.z,-b.w);
      // Cubic uses eased SLERP: orientation stops at knots, unlike PCHIP scalars.
      const [s,v,acc]=ease(mode==='cubic'?'smoothstep':mode,u);q=a.clone().slerp(b,s).normalize();
      const delta=b.clone().multiply(a.clone().invert()).normalize(),length=Math.hypot(delta.x,delta.y,delta.z),angle=2*Math.atan2(length,Math.max(0,delta.w));
      const axis=length>1e-12?[delta.x/length,delta.y/length,delta.z/length]:[0,0,0];
      omega=axis.map(x=>x*angle*v/h);alpha=axis.map(x=>x*angle*acc/(h*h));
      const e=new T.Euler().setFromQuaternion(q,'ZYX');rpy=[e.x,e.y,e.z];
    }else{
      const [,pitch,yaw]=rpy,[rd,pd,yd]=angles.map(c=>c[1]),[rdd,pdd,ydd]=angles.map(c=>c[2]),cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);
      omega=[cy*cp*rd-sy*pd,sy*cp*rd+cy*pd,yd-sp*rd];
      alpha=[cy*cp*rdd-sy*pdd+(-sy*yd*cp-cy*sp*pd)*rd-cy*yd*pd,
        sy*cp*rdd+cy*pdd+(cy*yd*cp-sy*sp*pd)*rd-sy*yd*pd,ydd-sp*rdd-cp*pd*rd];
    }
    return {time:t,base_position:position.map(c=>c[0]),base_rpy:rpy,base_quaternion:q.toArray(),base_linear_velocity:position.map(c=>c[1]),base_linear_acceleration:position.map(c=>c[2]),base_angular_velocity:omega,base_angular_acceleration:alpha,joint_positions:joints.map(c=>c[0]),joint_velocities:joints.map(c=>c[1]),joint_accelerations:joints.map(c=>c[2])};
  }
  function count(p,fps){
    if(!finite(fps)||fps<1||fps>1000)throw Error('Sample rate must be 1–1000 Hz');
    const n=Math.ceil(duration(p)*fps-1e-10)+1;
    if(n>100000||n*(30+p.model.joints.length*3)>5000000)throw Error('Too many samples; reduce duration or sample rate');return n;
  }
  async function sample(p,fps=p.fps){
    if(!p.keyframes.length&&!p.plan)throw Error('No keyframes');
    if(p.plan)fps=1/p.plan.settings.dt;
    const n=p.plan?p.plan.frames.length:count(p,fps),frames=[],end=duration(p),peak=p.model.joints.map(()=>0);let previous=null;
    for(let i=0;i<n;i++){
      const f=p.plan?clone(p.plan.frames[i]):evaluate(p,Math.min(i/fps,end)),q=f.base_quaternion;
      if(previous&&q.reduce((s,x,j)=>s+x*previous[j],0)<0)f.base_quaternion=q.map(x=>-x);
      previous=f.base_quaternion;frames.push(f);f.joint_velocities.forEach((v,j)=>peak[j]=Math.max(peak[j],Math.abs(v)));
      if(i%250===249)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return {schema:'urdf-viewer.reference.v1',name:p.name,fps,duration:end,metadata:{frame:p.reference?'fixed ground (Z up, ground at Z = 0)':'URDF root parent (before viewer up-axis rotation)',reference_transform:p.reference||null,position_unit:'m',angle_unit:'rad',quaternion_order:'xyzw',rpy_convention:'Rz(yaw) Ry(pitch) Rx(roll), unwrapped input',velocity_frame:p.reference?'fixed ground; angular velocity is spatial, not RPY rates':'root parent; angular velocity is spatial, not RPY rates',independent_joints:p.model.joints.map(j=>j.name),joint_units:p.model.joints.map(j=>j.type==='prismatic'?'m':'rad'),mimic:'Excluded; reproduce from URDF constraints',display_up_axis:p.model.up,ground_plane:p.reference?{normal:[0,0,1],offset:0}:p.model.groundPlane,urdf_ground_plane:p.model.groundPlane,kinematic_only:!p.plan,dynamics:p.plan?.report||null,optimization:p.optimization?.report||null,derivatives:p.plan?'Solver node derivatives; base rates expressed in the fixed ground frame. Inter-node display interpolation is not dynamically certified.':p.optimization?'Analytic C2 quintic splines; endpoint is exact. Jerk can jump at knots.':'Analytic, piecewise; linear velocities and cubic accelerations may jump at knots. Last row is the exact endpoint (possibly a shorter interval).',rotation:p.rotation,loop:p.loop,loop_seam_validated:false},diagnostics:{sampled_peak_joint_speeds:peak,velocity_limit_exceeded:p.model.joints.filter((j,i)=>finite(j.velocity)&&peak[i]>j.velocity+1e-8).map(j=>j.name)},frames};
  }
  function csv(data){
    const jointNames=data.metadata.independent_joints,fields=[['base_position',['x','y','z']],['base_rpy',['roll','pitch','yaw']],['base_quaternion',['x','y','z','w']],['base_linear_velocity',['x','y','z']],['base_angular_velocity',['x','y','z']],['base_linear_acceleration',['x','y','z']],['base_angular_acceleration',['x','y','z']],['joint_positions',jointNames],['joint_velocities',jointNames],['joint_accelerations',jointNames]];
    const quote=s=>'"'+String(s).replaceAll('"','""')+'"';
    const rows=[[quote('time_s'),...fields.flatMap(([name,axes])=>axes.map(axis=>quote(name+'.'+axis)))].join(',')];
    for(const f of data.frames)rows.push([f.time,...fields.flatMap(([name])=>f[name])].join(','));return rows.join('\n')+'\n';
  }
  return {modes,rotations,quaternion,duration,validate,evaluate,sample,csv,count,clone};
};
