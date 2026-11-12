/* Isolated URDF clone: optimization never moves the visible robot while solving. */
window.createTrajectoryKinematics=function(T,source,project){
  'use strict';
  const robot=source.clone(),joints=project.model.joints,geometry=new URDFStaticContacts(T),shapes=new Map();
  const ref=project.reference,origin=new T.Vector3(...(ref?.origin||[0,0,0])),rotation=new T.Quaternion(...(ref?.quaternion||[0,0,0,1]));
  const toGround=new T.Matrix4().compose(origin,rotation,new T.Vector3(1,1,1)).invert(),frame={toPlane:new T.Matrix4().makeRotationX(-Math.PI/2).multiply(toGround)};
  robot.updateMatrixWorld(true);
  for(const link of Object.values(robot.links)){
    const collision=link.children.filter(o=>o.isURDFCollider),visual=link.children.filter(o=>o.isURDFVisual),meshes=[];
    for(const group of collision.length?collision:visual)group.traverse(o=>{if(o.isMesh&&o.geometry?.attributes.position){if(!o.geometry.boundingBox)o.geometry.computeBoundingBox();meshes.push(o);}});
    if(meshes.length){const shape={link:link.name,object:link,meshes,fallback:!collision.length};geometry.prepare(shape);shapes.set(link.name,shape);}
  }
  function set(values){
    robot.position.fromArray(values);robot.position.applyQuaternion(rotation).add(origin);
    robot.quaternion.copy(rotation).multiply(new T.Quaternion().setFromEuler(new T.Euler(...values.slice(3,6),'ZYX')));
    joints.forEach((j,i)=>{robot.joints[j.name].ignoreLimits=false;robot.setJointValue(j.name,Math.max(j.lower??-Infinity,Math.min(j.upper??Infinity,values[i+6])));});robot.updateMatrixWorld(true);
  }
  function point(name){
    const shape=shapes.get(name);if(!shape)throw Error('No foot collision/visual geometry: '+name);
    const result=geometry.bottom(shape,frame);if(!result.points.length)throw Error('Invalid contact geometry: '+name);
    const mean=result.points.reduce((p,v)=>p.map((x,i)=>x+v[i]/result.points.length),[0,0,0]);return [mean[0],-mean[2],mean[1]];
  }
  function solve(matrix,rhs){
    const n=rhs.length,a=matrix.map((row,i)=>[...row,rhs[i]]);
    for(let i=0;i<n;i++){let pivot=i;for(let j=i+1;j<n;j++)if(Math.abs(a[j][i])>Math.abs(a[pivot][i]))pivot=j;[a[i],a[pivot]]=[a[pivot],a[i]];if(Math.abs(a[i][i])<1e-14)throw Error('Singular contact Jacobian');for(let j=i+1;j<n;j++){const ratio=a[j][i]/a[i][i];for(let k=i;k<=n;k++)a[j][k]-=ratio*a[i][k];}}
    const x=new Array(n);for(let i=n-1;i>=0;i--){let b=a[i][n];for(let j=i+1;j<n;j++)b-=a[i][j]*x[j];x[i]=b/a[i][i];}return x;
  }
  function projectPose(values,targets){
    const x=values.slice(),n=joints.length;
    for(let iteration=0;iteration<80;iteration++){
      set(x);const current=targets.flatMap(t=>point(t.name)),error=targets.flatMap((t,i)=>t.position.map((v,d)=>v-current[3*i+d]));
      if(Math.max(...error.map(Math.abs))<.0001)return x;
      const m=error.length,J=Array.from({length:m},()=>new Float64Array(n));
      for(let j=0;j<n;j++){
        const original=x[j+6],joint=joints[j],step=Number.isFinite(joint.upper)&&original+1e-5>joint.upper?-1e-5:1e-5;x[j+6]+=step;set(x);
        const shifted=targets.flatMap(t=>point(t.name));for(let i=0;i<m;i++)J[i][j]=(shifted[i]-current[i])/step;x[j+6]=original;
      }
      const A=Array.from({length:m},(_,i)=>Array.from({length:m},(_,j)=>J[i].reduce((s,v,k)=>s+v*J[j][k],i===j?1e-6:0))),dual=solve(A,error),delta=joints.map((_,j)=>dual.reduce((s,v,i)=>s+J[i][j]*v,0)),scale=Math.min(1,.15/Math.max(1e-10,...delta.map(Math.abs)));
      joints.forEach((joint,j)=>x[j+6]=Math.max(joint.lower??-Infinity,Math.min(joint.upper??Infinity,x[j+6]+delta[j]*scale)));
    }
    set(x);const residual=Math.max(...targets.map(t=>Math.hypot(...point(t.name).map((v,d)=>v-t.position[d]))));
    if(residual>.0005)throw Error('Unreachable contact ('+(residual*1000).toFixed(2)+' mm); adjust base pose, support selection or phase');return x;
  }
  function clearance(){let min=Infinity;for(const shape of shapes.values())min=Math.min(min,geometry.bottom(shape,frame).min);return min;}
  return {set,point,project:projectPose,clearance};
};
