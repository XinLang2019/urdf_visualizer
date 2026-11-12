/* Playback of solver nodes. Inter-node SLERP/linear display is not a dynamics solve. */
window.createWholeBodyTrajectoryCore=function(T){
  const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
  function validate(plan,model){
    if(!plan||plan.schema!=='urdf-viewer.whole-body.v1'||plan.modelHash!==model.descriptionHash||!Array.isArray(plan.jointNames)||JSON.stringify(plan.jointNames)!==JSON.stringify(model.joints.map(j=>j.name)))throw Error('Planner model or joint order mismatch; load the matching planning model');
    const r=plan.report;
    if(!r?.discreteDynamicsFeasible||![0,1].includes(r.status)||!Number.isFinite(r.maximumConstraintViolation)||r.maximumConstraintViolation<0||!Number.isFinite(r.tolerance)||r.tolerance<=0||r.maximumConstraintViolation>r.tolerance||!Number.isFinite(r.solveSeconds)||r.solveSeconds<0)throw Error('The solver has not produced a feasible converged trajectory');
    const s=plan.settings,fields=['dt','friction','gravity','jointSpeed','angularSpeed','torqueScale','swingClearance','bodyClearance','legClearance','selfMargin','groundMargin','initialHip','initialThigh','initialCalf','crouchThigh','targetX','targetY','targetZ','targetRoll','targetPitch','targetYaw','waypointRoll','waypointPitch','waypointYaw','waypointTolerance','waypointHeight','maxIterations','rounds','tolerance','wallTime'];
    if(!s||!['backflip','frontflip','jump','stand'].includes(s.motion)||fields.some(k=>!Number.isFinite(s[k]))||s.dt<=0||!s.phases||['stand','crouch','load','flight','landing','settle'].some(k=>!Number.isFinite(s.phases[k])||s.phases[k]<0)||['meshCollision','symmetry','waypointEnabled'].some(k=>typeof s[k]!=='boolean')||s.lockLandingJoints!==undefined&&typeof s.lockLandingJoints!=='boolean')throw Error('Invalid whole-body planner settings');
    if(!Array.isArray(plan.frames)||plan.frames.length<2||plan.frames.length>10000||plan.frames[0].time!==0)throw Error('Invalid whole-body trajectory frames');
    let previous=-1;
    for(const f of plan.frames){
      if(!Number.isFinite(f.time)||f.time<=previous||f.time>3600)throw Error('Invalid planner frame times');previous=f.time;
      for(const name of ['base_position','base_rpy','base_linear_velocity','base_angular_velocity','base_linear_acceleration','base_angular_acceleration'])if(!vector(f[name],3))throw Error('Invalid planned '+name);
      if(!vector(f.base_quaternion,4)||Math.abs(Math.hypot(...f.base_quaternion)-1)>1e-5)throw Error('Invalid planned quaternion');
      for(const name of ['joint_positions','joint_velocities','joint_accelerations','joint_torques'])if(!vector(f[name],model.joints.length))throw Error('Invalid planned '+name);
      f.joint_positions.forEach((v,i)=>{const j=model.joints[i];if(Number.isFinite(j.lower)&&v<j.lower-1e-5||Number.isFinite(j.upper)&&v>j.upper+1e-5)throw Error('Planned joint exceeds current model limits: '+j.name);});
      if(!f.contact_forces||Object.values(f.contact_forces).some(v=>!vector(v,3)))throw Error('Invalid planned contact forces');
    }
    return plan;
  }
  function evaluate(plan,time){
    const fs=plan.frames,t=Math.max(0,Math.min(fs.at(-1).time,time));let k=0;while(k<fs.length-2&&fs[k+1].time<=t)k++;
    const a=fs[k],b=fs[k+1],u=(t-a.time)/(b.time-a.time),out={time:t};
    for(const name of ['base_position','base_rpy','base_linear_velocity','base_angular_velocity','base_linear_acceleration','base_angular_acceleration','joint_positions','joint_velocities','joint_accelerations','joint_torques'])out[name]=a[name].map((v,i)=>v+(b[name][i]-v)*u);
    out.base_quaternion=new T.Quaternion(...a.base_quaternion).slerp(new T.Quaternion(...b.base_quaternion),u).normalize().toArray();
    out.contact_forces=Object.fromEntries([...new Set([...Object.keys(a.contact_forces),...Object.keys(b.contact_forces)])].map(n=>[n,[0,1,2].map(i=>(a.contact_forces[n]?.[i]||0)*(1-u)+(b.contact_forces[n]?.[i]||0)*u)]));
    if(time<0||time>fs.at(-1).time)for(const field of ['base_linear_velocity','base_angular_velocity','base_linear_acceleration','base_angular_acceleration','joint_velocities','joint_accelerations'])out[field].fill(0);
    return out;
  }
  return {validate,evaluate};
};
