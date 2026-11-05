/* A common slope frame and reversible rigid placement; joint coordinates stay untouched. */
(() => {
  'use strict';
  window.URDFStaticTerrain=class {
    constructor(THREE,viewer,grid) {
      Object.assign(this,{T:THREE,viewer,grid});this.robot=null;this.base=null;this.active=false;this.angle=0;this.heading=0;this.offset=null;this.fit=null;this.bodyAttitude=null;
      this.rotation=new THREE.Quaternion();this.toWorld=new THREE.Matrix4();this.toPlane=new THREE.Matrix4();this.normal=new THREE.Vector3(0,1,0);
      this.flatPlane=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2);
      this.patch=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({color:0x6ba3c8,transparent:true,opacity:.12,side:THREE.DoubleSide,depthWrite:false}));
      this.patch.raycast=()=>{};this.patch.visible=false;this.patch.userData.staticsTerrain=true;viewer.scene.add(this.patch);
    }
    reset(restore=true) {
      if(restore&&this.robot===this.viewer.robot&&this.base){this.base.decompose(this.robot.position,this.robot.quaternion,this.robot.scale);this.robot.updateMatrixWorld(true);}
      this.robot=null;this.base=null;this.active=false;this.angle=0;this.heading=0;this.offset=null;this.fit=null;this.bodyAttitude=null;
      this.rotation.identity();this.toWorld.identity();this.toPlane.identity();this.normal.set(0,1,0);
      this.grid.quaternion.identity();this.grid.position.x=this.grid.position.z=0;
      this.viewer.plane.quaternion.copy(this.flatPlane);this.viewer.plane.position.x=this.viewer.plane.position.z=0;this.patch.visible=false;
    }
    supportShapes(shapes,geometry,selected=null) {
      if(selected)return shapes.filter(s=>selected.has(s.link));
      // Keep the deepest geometric link on each terminal chain. A hip/body mesh
      // is not a support just because another foot is currently higher up.
      const leaves=shapes.filter(s=>!shapes.some(other=>{
        if(other===s)return false;for(let p=other.object.parent;p;p=p.parent)if(p===s.object)return true;return false;
      }));
      const named=leaves.filter(s=>/(^|[_\W])(foot|feet|toe|wheel)([_\W]|$)/i.test(s.link)||geometry.bottom(s).kind==='wheelMidplane');
      return named.length?named:leaves;
    }
    fitSupports(supports,geometry) {
      const T=this.T,up=new T.Vector3(0,1,0),probe={toPlane:new T.Matrix4()},rotation=new T.Quaternion();
      const evaluate=(u,v)=>{
        const normal=new T.Vector3(u,1,v).normalize();rotation.setFromUnitVectors(normal,up);probe.toPlane.makeRotationFromQuaternion(rotation);
        const heights=supports.map(s=>geometry.bottom(s,probe).min),mean=heights.reduce((s,x)=>s+x,0)/heights.length,residual=heights.map(x=>x-mean);
        return {u,v,normal,heights,residual,cost:residual.reduce((s,x)=>s+x*x,0),gap:Math.max(...heights)-Math.min(...heights)};
      };
      // The contact-height plane is a seed, not the answer: support points on
      // spheres and wheel rims move when the body attitude changes.
      const centers=supports.map(s=>{const b=geometry.bottom(s);return b.points.reduce((sum,p)=>sum.add(new T.Vector3(...p)),new T.Vector3()).multiplyScalar(1/b.points.length);});
      const mean=centers.reduce((s,p)=>s.add(p),new T.Vector3()).multiplyScalar(1/centers.length);
      let xx=0,xz=0,zz=0,xy=0,zy=0;
      for(const p of centers){const d=p.clone().sub(mean);xx+=d.x*d.x;xz+=d.x*d.z;zz+=d.z*d.z;xy+=d.x*d.y;zy+=d.z*d.y;}
      const regularizer=Math.max(xx+zz,1e-6)*1e-10,det=(xx+regularizer)*(zz+regularizer)-xz*xz;
      let best=evaluate(-(xy*(zz+regularizer)-zy*xz)/det,-(zy*(xx+regularizer)-xy*xz)/det);
      if(!Number.isFinite(best.cost)||best.normal.y<.2)best=evaluate(0,0);
      let iterations=0;
      for(;iterations<35&&best.gap>1e-8;iterations++) {
        const epsilon=1e-4,a=evaluate(best.u+epsilon,best.v),b=evaluate(best.u,best.v+epsilon);
        let aa=0,ab=0,bb=0,ar=0,br=0;
        for(let i=0;i<best.residual.length;i++){const x=(a.residual[i]-best.residual[i])/epsilon,z=(b.residual[i]-best.residual[i])/epsilon;aa+=x*x;ab+=x*z;bb+=z*z;ar+=x*best.residual[i];br+=z*best.residual[i];}
        const damping=Math.max(aa+bb,1e-8)*1e-8,determinant=(aa+damping)*(bb+damping)-ab*ab;
        let du=(-ar*(bb+damping)+br*ab)/determinant,dv=(-br*(aa+damping)+ar*ab)/determinant;
        const length=Math.hypot(du,dv);if(!Number.isFinite(length)||length<1e-10)break;
        if(length>.4){du*=.4/length;dv*=.4/length;}
        let improved=false;
        for(const step of [1,.5,.25,.125,.0625]){const next=evaluate(best.u+du*step,best.v+dv*step);if(next.normal.y>=.2&&next.cost<best.cost-1e-18){best=next;improved=true;break;}}
        if(!improved)break;
      }
      return {...best,iterations};
    }
    attitude(rotation,forward,normal) {
      const T=this.T,up=new T.Vector3(0,1,0).applyQuaternion(rotation),right=forward.clone().cross(new T.Vector3(0,1,0)).applyQuaternion(rotation),front=forward.clone().applyQuaternion(rotation),degrees=180/Math.PI;
      const angles=n=>({pitch:Math.asin(T.MathUtils.clamp(front.dot(n),-1,1))*degrees,roll:Math.atan2(-right.dot(n),up.dot(n))*degrees,tilt:Math.acos(T.MathUtils.clamp(up.dot(n),-1,1))*degrees});
      return {horizontal:angles(new T.Vector3(0,1,0)),slope:angles(normal)};
    }
    place(shapes,geometry,angle,heading,{selected=null,tolerance=.005}={}) {
      const T=this.T,robot=this.viewer.robot;if(!robot)return null;
      robot.updateMatrix();const previous=robot.matrix.clone();
      if(this.robot!==robot||!this.base){robot.updateMatrix();this.base=robot.matrix.clone();this.robot=robot;}
      this.base.decompose(robot.position,robot.quaternion,robot.scale);this.viewer.world.updateWorldMatrix(true,true);
      const restore=()=>{previous.decompose(robot.position,robot.quaternion,robot.scale);this.viewer.world.updateWorldMatrix(true,true);};
      const supports=this.supportShapes(shapes,geometry,selected),links=supports.map(s=>s.link);
      if(!supports.length){restore();this.fit={success:false,reason:'noSupports',links,maxGap:null};return null;}
      const fitted=this.fitSupports(supports,geometry);
      if(!Number.isFinite(fitted.gap)||fitted.gap>tolerance+1e-8){restore();this.fit={success:false,reason:'nonCoplanar',links,maxGap:fitted.gap,iterations:fitted.iterations};return null;}
      const bottoms=shapes.map(s=>geometry.bottom(s)).filter(b=>Number.isFinite(b.min));
      const flatGround=Math.min(...bottoms.map(b=>b.min)),center=robot.getWorldPosition(new T.Vector3()),pivot=new T.Vector3(center.x,flatGround,center.z);
      const phi=heading*Math.PI/180,direction=new T.Vector3(Math.cos(phi),0,Math.sin(phi)),axis=direction.clone().cross(new T.Vector3(0,1,0));
      const slopeRotation=new T.Quaternion().setFromAxisAngle(axis,angle*Math.PI/180),normal=new T.Vector3(0,1,0).applyQuaternion(slopeRotation);
      const correction=new T.Quaternion().setFromUnitVectors(fitted.normal,new T.Vector3(0,1,0)),bodyRotation=slopeRotation.clone().multiply(correction);
      const forward=new T.Vector3(1,0,0).transformDirection(robot.matrixWorld);forward.y=0;if(forward.lengthSq()<1e-8)forward.set(0,0,1);forward.normalize();
      const transform=new T.Matrix4().makeTranslation(...pivot.toArray()).multiply(new T.Matrix4().makeRotationFromQuaternion(bodyRotation)).multiply(new T.Matrix4().makeTranslation(...pivot.clone().negate().toArray()));
      const placed=robot.parent.matrixWorld.clone().invert().multiply(transform).multiply(robot.matrixWorld);
      placed.decompose(robot.position,robot.quaternion,robot.scale);this.viewer.world.updateWorldMatrix(true,true);
      const toWorld=new T.Matrix4().makeRotationFromQuaternion(slopeRotation),frame={toPlane:toWorld.clone().invert()},offset=pivot.dot(normal);
      const supportBottoms=supports.map(s=>geometry.bottom(s,frame)),min=Math.min(...supportBottoms.map(b=>b.min));
      // Settle along the normal after attitude fitting; do not move endpoints
      // individually or overwrite the configured joint angles.
      const worldPosition=robot.getWorldPosition(new T.Vector3()).addScaledVector(normal,offset-min);robot.position.copy(robot.parent.worldToLocal(worldPosition));this.viewer.world.updateWorldMatrix(true,true);
      const finalBottoms=shapes.map(s=>geometry.bottom(s,frame)),contactHeights=finalBottoms.filter(b=>links.includes(b.link)).map(b=>b.min),maxGap=Math.max(...contactHeights.map(y=>Math.abs(y-offset)));
      const penetrating=finalBottoms.filter(b=>b.min<offset-tolerance).map(b=>b.link);
      if(maxGap>tolerance+1e-8||penetrating.length){restore();this.fit={success:false,reason:penetrating.length?'bodyPenetration':'nonCoplanar',links,maxGap,penetrating};return null;}
      this.rotation.copy(slopeRotation);this.toWorld.copy(toWorld);this.toPlane.copy(frame.toPlane);this.normal.copy(normal);
      this.active=true;this.angle=angle;this.heading=heading;this.offset=offset;
      this.fit={success:true,links,maxGap,iterations:fitted.iterations};this.bodyAttitude=this.attitude(bodyRotation,forward,normal);
      const box=new T.Box3().setFromObject(robot),size=box.getSize(new T.Vector3()).length();
      this.patch.scale.setScalar(Math.max(1,size*1.8));this.patch.position.copy(pivot);
      this.sync(this.offset);return this.offset;
    }
    sync(offset=this.offset) {
      if(!this.active||this.robot!==this.viewer.robot||offset===null)return;
      this.offset=offset;this.grid.quaternion.copy(this.rotation);this.grid.position.copy(this.normal).multiplyScalar(offset+.001);
      this.viewer.plane.quaternion.copy(this.rotation).multiply(this.flatPlane);this.viewer.plane.position.copy(this.normal).multiplyScalar(offset-.001);
      this.patch.quaternion.copy(this.viewer.plane.quaternion);this.patch.position.addScaledVector(this.normal,offset-this.patch.position.dot(this.normal)-.002);this.patch.visible=true;
    }
    world(point){return new this.T.Vector3(...point).applyMatrix4(this.toWorld);}
    plane(point){return point.clone().applyMatrix4(this.toPlane);}
    onPlane(x,z,offset){return this.world([x,offset,z]);}
    projectGravity(point,offset) {const p=point.clone();p.y+=(offset-this.normal.dot(p))/this.normal.y;return p;}
    get snapshot(){return {active:this.active,angle:this.angle,heading:this.heading,normal:this.normal.toArray(),offset:this.offset,fit:this.fit,bodyAttitude:this.bodyAttitude};}
  };
})();
