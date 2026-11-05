/* Geometric snap features. Coordinates stay in mesh space so joints can move them. */
(() => {
  'use strict';
  window.URDFMeasurementFeatures = class {
    constructor(THREE) { this.THREE=THREE;this.cache=new WeakMap(); }
    get(geometry) {
      if(!this.cache.has(geometry))this.cache.set(geometry,this.extract(geometry));
      return this.cache.get(geometry);
    }
    async extract(geometry) {
      const T=this.THREE,V=(x=0,y=0,z=0)=>new T.Vector3(x,y,z),points=[],circles=[];
      const point=(position,kind,fitted=false)=>points.push({position,kind,fitted});
      const circle=(center,normal,radius,fitted=false,neighbors=new Set())=>{
        const existing=circles.find(c=>c.center.distanceTo(center)<tolerance*4 && Math.abs(c.radius-radius)<tolerance*4 && Math.abs(c.normal.dot(normal))>.999);
        if(existing){for(const id of neighbors)existing.neighbors.add(id);return;}
        circles.push({center,normal,radius,fitted,neighbors});point(center.clone(),'circleCenter',fitted);
      };
      geometry.computeBoundingBox();
      if(!geometry.boundingBox || geometry.boundingBox.isEmpty())return {points,circles};
      const extent=geometry.boundingBox.getSize(V()).length(),tolerance=Math.max(extent*1e-6,1e-9),p=geometry.parameters;
      if(geometry.type==='BoxGeometry' && p) {
        const half=V(p.width/2,p.height/2,p.depth/2);
        point(V(),'geometryCenter');
        for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])point(V(x,y,z).multiply(half),'rectangleVertex');
        for(let axis=0;axis<3;axis++)for(const sign of [-1,1])point(V().setComponent(axis,half.getComponent(axis)*sign),'rectangleCenter');
        return {points,circles};
      }
      if(geometry.type==='SphereGeometry' && p) {
        point(V(),'sphereCenter');return {points,circles};
      }
      if(geometry.type==='CylinderGeometry' && p && Math.abs(p.thetaLength-Math.PI*2)<1e-6 && p.radiusTop===p.radiusBottom) {
        point(V(),'geometryCenter');
        for(const y of [-p.height/2,0,p.height/2])circle(V(0,y,0),V(0,1,0),p.radiusTop);
        return {points,circles};
      }
      // Weld STL's duplicate vertices, then find connected planar patches. Work is
      // cooperative and cached per geometry; complex links never block a full scan.
      const pause=()=>new Promise(resolve=>setTimeout(resolve,0));
      const attribute=geometry.getAttribute('position');
      if(!attribute)return {points,circles};
      const vertices=[],vertexIds=new Uint32Array(attribute.count),welded=new Map();
      for(let i=0;i<attribute.count;i++) {
        const v=V().fromBufferAttribute(attribute,i),key=[v.x,v.y,v.z].map(n=>Math.round(n/tolerance)).join(',');
        let id=welded.get(key);if(id===undefined){id=vertices.length;vertices.push(v);welded.set(key,id);}vertexIds[i]=id;
        if(i%6000===5999)await pause();
      }
      const faces=[],edges=new Map(),count=geometry.index?.count ?? attribute.count;
      for(let i=0;i+2<count;i+=3) {
        const ids=[0,1,2].map(j=>vertexIds[geometry.index?geometry.index.getX(i+j):i+j]);
        const [a,b,c]=ids.map(id=>vertices[id]),normal=b.clone().sub(a).cross(c.clone().sub(a));
        if(normal.lengthSq()<tolerance**4)continue;
        normal.normalize();
        const face={ids,normal,plane:normal.dot(a),edges:[],group:-1},id=faces.length;faces.push(face);
        for(let j=0;j<3;j++) {
          const a=ids[j],b=ids[(j+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
          let edge=edges.get(key);if(!edge){edge={a,b,faces:[]};edges.set(key,edge);}edge.faces.push(id);face.edges.push(edge);
        }
        if(i%12000===11997)await pause();
      }
      const groups=[];
      for(let i=0;i<faces.length;i++) {
        if(faces[i].group!==-1)continue;
        const id=groups.length,seed=faces[i],stack=[i],members=[];seed.group=id;
        while(stack.length) {
          const current=stack.pop();members.push(current);
          for(const edge of faces[current].edges)for(const next of edge.faces) {
            const face=faces[next];
            if(face.group===-1 && seed.normal.dot(face.normal)>.99999 && Math.abs(seed.plane-face.plane)<tolerance*8){face.group=id;stack.push(next);}
          }
          if(members.length%6000===0)await pause();
        }
        groups.push(members);
        if(i%3000===0)await pause();
      }
      function loops(boundary) {
        const adjacency=new Map(),remaining=new Set(boundary),result=[];
        for(const edge of boundary)for(const id of [edge.a,edge.b]){if(!adjacency.has(id))adjacency.set(id,[]);adjacency.get(id).push(edge);}
        for(const first of boundary) {
          if(!remaining.has(first))continue;
          const path=[],neighbors=new Set();let edge=first,id=first.a,closed=false;
          while(remaining.has(edge)) {
            remaining.delete(edge);path.push(vertices[id]);
            for(const f of edge.faces)neighbors.add(faces[f].group);
            id=edge.a===id?edge.b:edge.a;
            if(id===first.a){closed=true;break;}
            const options=adjacency.get(id);
            if(options.length!==2)break;
            edge=options.find(e=>remaining.has(e));if(!edge)break;
          }
          if(closed && path.length>=3)result.push({path,neighbors});
        }
        return result;
      }
      const fitCircle=path=>{
        if(path.length<8)return null;
        const mean=path.reduce((sum,p)=>sum.add(p),V()).multiplyScalar(1/path.length);
        const u=path[0].clone().sub(mean).normalize();
        const normal=u.clone().cross(path[Math.floor(path.length/3)].clone().sub(mean)).normalize(),v=normal.clone().cross(u);
        if(normal.lengthSq()<.99)return null;
        let xx=0,yy=0,xy=0,xb=0,yb=0;
        for(const p of path) {
          const d=p.clone().sub(mean),x=d.dot(u),y=d.dot(v),b=(x*x+y*y)/2;
          if(Math.abs(d.dot(normal))>tolerance*12)return null;
          xx+=x*x;yy+=y*y;xy+=x*y;xb+=x*b;yb+=y*b;
        }
        const determinant=xx*yy-xy*xy;if(determinant<tolerance**4)return null;
        const center=mean.clone().addScaledVector(u,(xb*yy-yb*xy)/determinant).addScaledVector(v,(yb*xx-xb*xy)/determinant);
        const radius=path.reduce((sum,p)=>sum+p.distanceTo(center),0)/path.length;
        if(radius<tolerance*10 || path.some(p=>Math.abs(p.distanceTo(center)-radius)>Math.max(tolerance*12,radius*.006)))return null;
        const angles=path.map(p=>{const d=p.clone().sub(center);return Math.atan2(d.dot(v),d.dot(u));}).sort((a,b)=>a-b);
        if(angles.some((a,i)=>(i===angles.length-1?angles[0]+Math.PI*2:angles[i+1])-a>Math.PI/3))return null;
        return {center,normal,radius};
      };
      const inspectLoop=({path,neighbors})=>{
        const ring=fitCircle(path);
        if(ring){circle(ring.center,ring.normal,ring.radius,true,neighbors);return;}
        // Remove collinear tessellation vertices before testing an actual rectangle.
        const corners=path.filter((p,i)=>{
          const a=p.clone().sub(path[(i+path.length-1)%path.length]).normalize(),b=path[(i+1)%path.length].clone().sub(p).normalize();
          return a.dot(b)<.9999;
        });
        if(corners.length!==4)return;
        const sides=corners.map((p,i)=>corners[(i+1)%4].clone().sub(p));
        if(sides.some((s,i)=>Math.abs(s.clone().normalize().dot(sides[(i+1)%4].clone().normalize()))>.001))return;
        if(corners[0].clone().add(corners[2]).distanceTo(corners[1].clone().add(corners[3]))>tolerance*12)return;
        for(const p of corners)point(p.clone(),'rectangleVertex',true);
        point(corners[0].clone().add(corners[2]).multiplyScalar(.5),'rectangleCenter',true);
      };
      for(let id=0;id<groups.length;id++) {
        const group=groups[id];if(group.length<2)continue;
        const boundary=new Set();
        for(const f of group)for(const edge of faces[f].edges)if(edge.faces.length===1 || edge.faces.some(other=>faces[other].group!==id))boundary.add(edge);
        for(const loop of loops([...boundary]))inspectLoop(loop);
        if(id%120===0)await pause();
      }
      // Open cylinder rims are boundaries even when no cap triangles exist.
      for(const loop of loops([...edges.values()].filter(e=>e.faces.length===1)))inspectLoop(loop);
      // Material seams and tiny deviations can split one cap into several planar
      // patches. Sharp outline loops still recover its complete circular rim.
      const sharp=[...edges.values()].filter(edge=>edge.faces.length===1 || (edge.faces.length===2 && faces[edge.faces[0]].normal.dot(faces[edge.faces[1]].normal)<.85));
      for(const loop of loops(sharp)) {
        const ring=fitCircle(loop.path);if(ring)circle(ring.center,ring.normal,ring.radius,true,loop.neighbors);
      }
      const rims=circles.slice();
      for(let i=0;i<rims.length;i++)for(let j=i+1;j<rims.length;j++) {
        const a=rims[i],b=rims[j],d=b.center.clone().sub(a.center),height=Math.abs(d.dot(a.normal));
        if(Math.abs(a.radius-b.radius)>tolerance*12 || Math.abs(a.normal.dot(b.normal))<.99999 || height<tolerance*12 || d.clone().cross(a.normal).length()>tolerance*12)continue;
        // Two unrelated coaxial holes are not enough: their rims must share side patches.
        if([...a.neighbors].filter(id=>b.neighbors.has(id)).length<4)continue;
        const center=a.center.clone().add(b.center).multiplyScalar(.5);
        point(center.clone(),'geometryCenter',true);circle(center,a.normal.clone(),a.radius,true);
      }
      // A mesh's bounds are useful, but are explicitly distinct from a geometric center.
      point(geometry.boundingBox.getCenter(V()),'boundsCenter');
      return {points,circles};
    }
    worldCandidates(mesh,features) {
      const T=this.THREE,result=[],keys=new Set(),matrix=mesh.matrixWorld,inverse=matrix.clone().invert(),normalMatrix=new T.Matrix3().getNormalMatrix(matrix);
      const add=(local,kind,fitted=false,angle=null)=>{
        const world=local.clone().applyMatrix4(matrix),key=world.toArray().map(n=>n.toFixed(8)).join(',');
        if(keys.has(key))return;keys.add(key);result.push({object:mesh,local:local.clone(),world,kind,fitted,angle});
      };
      for(const p of features.points)add(p.position,p.kind,p.fitted);
      for(const ring of features.circles) {
        const normal=ring.normal.clone().applyMatrix3(normalMatrix).normalize();
        const major=[0,1,2].sort((a,b)=>Math.abs(normal.getComponent(b))-Math.abs(normal.getComponent(a)))[0];
        if(normal.getComponent(major)<0)normal.negate();
        // Zero degrees follows world +X projected onto the circle; use +Y when parallel.
        let u=new T.Vector3(1,0,0).addScaledVector(normal,-normal.x);
        if(u.lengthSq()<1e-8)u.set(0,1,0).addScaledVector(normal,-normal.y);
        u.normalize();const v=normal.clone().cross(u);
        for(let i=0;i<4;i++) {
          const direction=(i%2?v:u).clone().multiplyScalar(i<2?1:-1).transformDirection(inverse);
          add(ring.center.clone().addScaledVector(direction,ring.radius),'quadrant',ring.fitted,i*90);
        }
      }
      return result;
    }
  };
})();
