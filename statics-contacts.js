/* Collision support geometry. Wheels use one equivalent contact in the tread midplane. */
(function(){
  'use strict';
  window.URDFStaticContacts=class {
    constructor(THREE){this.T=THREE;}
    prepare(shape) {
      const T=this.T,link=shape.object;
      shape.wheel=null;
      // Spherical point feet keep their exact support point. Native primitives are
      // handled analytically below; infer an axle only for imported wheel meshes.
      if(shape.meshes.every(m=>['SphereGeometry','CylinderGeometry','BoxGeometry'].includes(m.geometry.type)))return;
      let hasMovingChild=false;
      link.traverse(o=>{if(o!==link&&o.isURDFJoint&&o.jointType!=='fixed')hasMovingChild=true;});
      if(hasMovingChild)return;
      let axle=link.parent;
      while(axle&&!(axle.isURDFJoint&&['revolute','continuous'].includes(axle.jointType)))axle=axle.parent;
      if(!axle?.axis?.lengthSq())return;
      const inverse=link.matrixWorld.clone().invert(),axis=axle.axis.clone().transformDirection(axle.matrixWorld).transformDirection(inverse);
      const origin=axle.getWorldPosition(new T.Vector3()).applyMatrix4(inverse),points=[];
      let radius=0,minAxial=Infinity,maxAxial=-Infinity;
      for(const mesh of shape.meshes) {
        const transform=inverse.clone().multiply(mesh.matrixWorld),attribute=mesh.geometry.attributes.position,p=new T.Vector3();
        for(let i=0;i<attribute.count;i++) {
          p.fromBufferAttribute(attribute,i).applyMatrix4(transform).sub(origin);
          const axial=p.dot(axis),radial=p.clone().addScaledVector(axis,-axial),r=radial.length();
          radius=Math.max(radius,r);minAxial=Math.min(minAxial,axial);maxAxial=Math.max(maxAxial,axial);points.push({axial,radial,r});
        }
      }
      if(radius<1e-8||maxAxial-minAxial>radius*1.8)return;
      const rim=points.filter(p=>p.r>=radius*.95);
      let lo=Infinity,hi=-Infinity;for(const p of rim){lo=Math.min(lo,p.axial);hi=Math.max(hi,p.axial);}
      if(hi-lo<radius*.01||hi-lo>radius*1.5)return;
      let u=new T.Vector3(1,0,0).addScaledVector(axis,-axis.x);
      if(u.lengthSq()<1e-8)u.set(0,1,0).addScaledVector(axis,-axis.y);
      u.normalize();const v=axis.clone().cross(u);
      const angles=[...new Set(rim.map(p=>Math.atan2(p.radial.dot(v),p.radial.dot(u)).toFixed(5)))].map(Number).sort((a,b)=>a-b);
      // A rectangular foot has only isolated outer corners, not a circular tread.
      if(angles.length<12||angles.some((a,i)=>(i+1<angles.length?angles[i+1]:angles[0]+2*Math.PI)-a>Math.PI/3))return;
      shape.wheel={center:origin.addScaledVector(axis,(lo+hi)/2),axis,radius};
    }
    circleBottom(center,axis,radius,matrix) {
      const T=this.T,e=matrix.elements;
      // M^T * gravity projects world down into the circle's local radial plane.
      const direction=new T.Vector3(-e[1],-e[5],-e[9]);
      direction.addScaledVector(axis,-direction.dot(axis));
      if(direction.lengthSq()<1e-16)return null; // Upright cylinders retain their flat cap patch.
      return center.clone().addScaledVector(direction.normalize(),radius).applyMatrix4(matrix);
    }
    bottom(shape,frame=null) {
      const T=this.T,transform=matrix=>frame?frame.toPlane.clone().multiply(matrix):matrix;
      if(shape.wheel) {
        const w=shape.wheel,p=this.circleBottom(w.center,w.axis,w.radius,transform(shape.object.matrixWorld));
        if(p)return {link:shape.link,min:p.y,points:[p.toArray()],fallback:shape.fallback,kind:'wheelMidplane'};
      }
      // Prefer analytic wheel primitives even when visual geometry is the fallback.
      if(shape.fallback&&!shape.meshes.every(m=>['SphereGeometry','CylinderGeometry'].includes(m.geometry.type))) {
        const box=new T.Box3();for(const mesh of shape.meshes)box.union(mesh.geometry.boundingBox.clone().applyMatrix4(transform(mesh.matrixWorld)));
        const c=box.getCenter(new T.Vector3());return {link:shape.link,min:box.min.y,points:[[c.x,box.min.y,c.z]],fallback:true};
      }
      let min=Infinity,points=[],wheel=false;
      const consider=(p,isWheel=false)=>{
        if(p.y<min-1e-5){min=p.y;points=[p.toArray()];wheel=isWheel;}
        else if(p.y<=min+1e-5){points.push(p.toArray());wheel=wheel||isWheel;}
      };
      for(const mesh of shape.meshes) {
        const geom=mesh.geometry,matrix=transform(mesh.matrixWorld),parameters=geom.parameters;
        if(geom.type==='SphereGeometry') {
          const e=matrix.elements,d=new T.Vector3(-e[1],-e[5],-e[9]).normalize().multiplyScalar(parameters.radius);
          consider(d.applyMatrix4(matrix));
        }else {
          if(geom.type==='CylinderGeometry'&&parameters.radiusTop===parameters.radiusBottom&&Math.abs(parameters.thetaLength-2*Math.PI)<1e-6) {
            const cylinderAxis=new T.Vector3(0,1,0).transformDirection(mesh.matrixWorld);
            let axle=shape.object.parent;while(axle&&!(axle.isURDFJoint&&['revolute','continuous'].includes(axle.jointType)))axle=axle.parent;
            const alignedAxle=axle?.axis&&Math.abs(cylinderAxis.dot(axle.axis.clone().transformDirection(axle.matrixWorld)))>1-1e-5;
            const radialCylinder=Math.abs(new T.Vector3(0,1,0).transformDirection(matrix).y)<.8;
            // A cylinder aligned with its wheel joint uses the requested tread
            // midplane. An ankle's cylindrical sole instead keeps its cap/rim.
            const p=(alignedAxle||!axle&&radialCylinder)?this.circleBottom(new T.Vector3(),new T.Vector3(0,1,0),parameters.radiusTop,matrix):null;
            if(p){consider(p,true);continue;}
          }
          const attribute=geom.attributes.position,p=new T.Vector3();
          for(let i=0;i<attribute.count;i++){p.fromBufferAttribute(attribute,i).applyMatrix4(matrix);consider(p);}
        }
      }
      const hull=window.URDFStatics.hull2(points),chosen=hull.length<=4?hull:[...new Set([0,1,2,3].map(q=>{
        const angle=q*Math.PI/2;let index=0,best=-Infinity;
        for(let i=0;i<hull.length;i++){const score=hull[i][0]*Math.cos(angle)+hull[i][1]*Math.sin(angle);if(score>best){best=score;index=i;}}
        return index;
      }))].map(i=>hull[i]);
      return {link:shape.link,min,points:chosen.map(p=>[p[0],min,p[1]]),fallback:shape.fallback,kind:wheel?'wheelMidplane':'surface'};
    }
  };
})();
