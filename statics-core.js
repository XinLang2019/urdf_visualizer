/* Quasi-static floating-base analysis with unilateral point contacts and Coulomb static friction on a common plane.
 * Units: m, kg, s, N, N·m. No renderer or DOM dependencies.
 */
(function(root){
  'use strict';
  const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,s)=>a.map(v=>v*s);
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm=a=>Math.hypot(...a),zero=()=>[0,0,0];
  function pseudoInverse(a) {
    a=a.map(r=>r.slice());const size=a.length,v=Array.from({length:size},(_,i)=>Array.from({length:size},(_,j)=>Number(i===j)));
    const pairs=[];for(let i=0;i<size;i++)for(let j=i+1;j<size;j++)pairs.push([i,j]);
    for(let n=0;n<size*size*12;n++) {
      let p=0,q=1;for(const [i,j]of pairs)if(Math.abs(a[i][j])>Math.abs(a[p][q])){p=i;q=j;}
      if(Math.abs(a[p][q])<1e-14)break;
      const t=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(t),s=Math.sin(t),x=a[p][p],y=a[q][q],z=a[p][q];
      a[p][p]=c*c*x-2*s*c*z+s*s*y;a[q][q]=s*s*x+2*s*c*z+c*c*y;a[p][q]=a[q][p]=0;
      for(let k=0;k<size;k++) {
        if(k!==p&&k!==q){const x=a[k][p],y=a[k][q];a[k][p]=a[p][k]=c*x-s*y;a[k][q]=a[q][k]=s*x+c*y;}
        const x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;
      }
    }
    const threshold=Math.max(...a.map((r,i)=>r[i]),1)*1e-12;
    return Array.from({length:size},(_,i)=>Array.from({length:size},(_,j)=>v[i].reduce((s,x,k)=>s+(a[k][k]>threshold?x*v[j][k]/a[k][k]:0),0)));
  }
  const matVec=(a,b)=>a.map(r=>dot(r,b));
  function hull2(points) {
    const sorted=points.map(p=>[p[0],p[2]]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]).filter((p,i,a)=>!i||norm(sub(p,a[i-1]))>1e-9);
    if(sorted.length<3)return sorted;
    const turn=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const half=list=>{const h=[];for(const p of list){while(h.length>1&&turn(h[h.length-2],h[h.length-1],p)<=1e-12)h.pop();h.push(p);}return h;};
    const lo=half(sorted),hi=half(sorted.slice().reverse());lo.pop();hi.pop();return lo.concat(hi);
  }
  function supportMargin(hull,com) {
    const p=[com[0],com[2]],distance=(a,b)=>{const d=sub(b,a),s=dot(d,d),t=s?Math.max(0,Math.min(1,dot(sub(p,a),d)/s)):0;return norm(sub(p,add(a,mul(d,t))));};
    if(!hull.length)return -Infinity;
    if(hull.length===1)return -norm(sub(p,hull[0]));
    if(hull.length===2)return -distance(hull[0],hull[1]);
    let margin=Infinity;
    for(let i=0;i<hull.length;i++){const a=hull[i],b=hull[(i+1)%hull.length],d=sub(b,a);margin=Math.min(margin,(d[0]*(p[1]-a[1])-d[1]*(p[0]-a[0]))/norm(d));}
    return margin;
  }
  function contactLoads(points,com,weight) {
    if(!points.length)return {valid:false,reason:'noContacts',forces:[],hull:[],margin:null};
    const hull=hull2(points),margin=supportMargin(hull,com),tolerance=1e-7;
    if(margin < -tolerance)return {valid:false,reason:'outsideSupport',forces:[],hull,margin};
    // Center and scale to keep equilibrium well-conditioned across model sizes.
    const scale=Math.max(...points.map(p=>Math.hypot(p[0]-com[0],p[2]-com[2])),.01);
    const columns=points.map(p=>[1,(p[0]-com[0])/scale,(p[2]-com[2])/scale]);
    const gram=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>columns.reduce((s,c)=>s+c[i]*c[j],0)));
    const inverse=pseudoInverse(gram),b=[1,0,0],count=points.length;
    // Dykstra projects zero onto {A w=b} ∩ {w>=0}: the unique minimum L2 load.
    let w=Array(count).fill(0),p=w.slice(),q=w.slice(),residual=Infinity,delta=Infinity,iterations=0;
    for(;iterations<4000;iterations++) {
      const v=w.map((x,i)=>x+p[i]),av=columns.reduce((s,c,i)=>add(s,mul(c,v[i])),zero()),lambda=matVec(inverse,sub(b,av));
      const y=v.map((x,i)=>x+dot(columns[i],lambda));p=v.map((x,i)=>x-y[i]);
      const z=y.map((x,i)=>x+q[i]),next=z.map(x=>Math.max(0,x));q=z.map((x,i)=>x-next[i]);
      delta=Math.max(...next.map((x,i)=>Math.abs(x-w[i])));w=next;
      residual=norm(sub(columns.reduce((s,c,i)=>add(s,mul(c,w[i])),zero()),b));
      if(residual<1e-9&&delta<1e-10)break;
    }
    if(residual>1e-6||delta>1e-6)return {valid:false,reason:'solverFailed',forces:[],hull,margin,residual,iterations};
    return {valid:true,reason:margin>tolerance&&hull.length>=3?'balanced':'marginal',forces:w.map(x=>x*weight),hull,margin,residual,iterations};
  }
  // Minimum squared 3D contact loads subject to six equilibrium equations and
  // circular Coulomb cones. Forces are normalized by weight for scale invariance.
  function frictionLoads(points,com,weight,normal,friction) {
    const planar=contactLoads(points,com,weight),n=mul(normal,1/norm(normal));
    const requiredFriction=Math.hypot(n[0],n[2])/n[1];
    const tangent=mul(sub([1,0,0],mul(n,n[0])),1/Math.hypot(n[1],n[2])),side=cross(n,tangent),basis=[n,tangent,side];
    const offset=points.length?dot(n,points[0]):0,projection=[com[0],(offset-n[0]*com[0]-n[2]*com[2])/n[1],com[2]];
    const surfacePoints=points.map(p=>[dot(p,tangent),0,dot(p,side)]),surfaceCom=[dot(projection,tangent),0,dot(projection,side)];
    const common={...planar,margin:points.length?supportMargin(hull2(surfacePoints),surfaceCom):null,normal:n,friction,requiredFriction};
    if(!planar.valid)return common;
    if(!(friction>=0)||!Number.isFinite(friction)||n[1]<=1e-6)return {...common,valid:false,reason:'invalidTerrain',forces:[]};
    if(friction+1e-9<requiredFriction)return {...common,valid:false,reason:'sliding',forces:[]};
    if(friction-requiredFriction<1e-7)return {...common,forceVectors:planar.forces.map(f=>[0,f,0]),distribution:'minimumNorm'};
    const scale=Math.max(...points.map(p=>norm(sub(p,com))),.01);
    const columns=points.flatMap(p=>basis.map(f=>[...f,...mul(cross(sub(p,com),f),1/scale)]));
    const gram=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>columns.reduce((s,c)=>s+c[i]*c[j],0))),inverse=pseudoInverse(gram),b=[0,1,0,0,0,0];
    const apply=w=>columns.reduce((s,c,i)=>add(s,mul(c,w[i])),Array(6).fill(0));
    const cone=v=>{
      const [a,x,y]=v,r=Math.hypot(x,y);
      if(a>=0&&r<=friction*a)return v;
      if(a+friction*r<=0)return [0,0,0];
      const normal=(a+friction*r)/(1+friction*friction),ratio=r?friction*normal/r:0;
      return [normal,x*ratio,y*ratio];
    };
    let w=Array(columns.length).fill(0),p=w.slice(),q=w.slice(),residual=Infinity,delta=Infinity,iterations=0;
    for(;iterations<6000;iterations++) {
      const v=add(w,p),lambda=matVec(inverse,sub(b,apply(v))),y=v.map((x,i)=>x+dot(columns[i],lambda));p=sub(v,y);
      const z=add(y,q),next=[];for(let i=0;i<z.length;i+=3)next.push(...cone(z.slice(i,i+3)));
      q=sub(z,next);delta=Math.max(...next.map((x,i)=>Math.abs(x-w[i])));w=next;residual=norm(sub(apply(w),b));
      if(residual<1e-9&&delta<1e-10)break;
    }
    // The proportional vertical solution is an exactly feasible witness on a
    // common plane when mu >= tan(slope), including singular/limiting contacts.
    const converged=residual<1e-7&&delta<1e-7;
    const vectors=points.map((_,i)=>converged?basis.reduce((s,f,k)=>add(s,mul(f,w[i*3+k]*weight)),zero()):[0,planar.forces[i],0]);
    return {...common,forceVectors:vectors,forces:vectors.map(norm),iterations,residual,distribution:converged?'minimumNorm':'proportionalFeasible'};
  }
  function analyze({masses,joints,contacts,gravity=9.81,groundNormal=[0,1,0],friction=0}) {
    const mass=masses.reduce((s,m)=>s+m.mass,0);
    if(!(mass>0)||!Number.isFinite(mass)||!Number.isFinite(gravity)||gravity<=0)return {valid:false,reason:'noMass',mass:0,joints:[],contacts:[]};
    const com=mul(masses.reduce((s,m)=>add(s,mul(m.position,m.mass)),zero()),1/mass),weight=mass*gravity;
    const length=norm(groundNormal);
    if(!Number.isFinite(length)||length<1e-8||groundNormal[1]<=0||!Number.isFinite(friction)||friction<0)return {valid:false,reason:'invalidTerrain',mass,com,weight,gravity,joints:[],contacts:[]};
    const normal=mul(groundNormal,1/length),inclined=Math.hypot(normal[0],normal[2])>1e-8;
    const solution=inclined?frictionLoads(contacts.map(c=>c.position),com,weight,normal,friction):contactLoads(contacts.map(c=>c.position),com,weight);
    const loadedContacts=contacts.map((c,i)=>{
      if(!solution.valid)return {...c,force:null};
      const forceVector=solution.forceVectors?.[i]||[0,solution.forces[i],0],normalForce=Math.max(0,dot(forceVector,normal)),normalVector=mul(normal,normalForce),frictionVector=sub(forceVector,normalVector),frictionForce=norm(frictionVector),frictionLimit=friction*normalForce;
      return {...c,force:norm(forceVector),forceVector,normalForce,normalVector,frictionForce,frictionVector,frictionLimit,frictionUtilization:frictionLimit>1e-9?frictionForce/frictionLimit:frictionForce>1e-8?Infinity:0};
    });
    const result={...solution,mass,com,weight,gravity,groundNormal:normal,friction,requiredFriction:Math.hypot(normal[0],normal[2])/normal[1],contacts:loadedContacts,joints:[]};
    if(!solution.valid)return result;
    const byLink=new Map();
    function bucket(name){if(!byLink.has(name))byLink.set(name,{mass:0,weightedCom:zero(),gravityForce:zero(),gravityMoment:zero(),supportForce:zero(),supportMoment:zero(),normalForce:zero(),normalMoment:zero(),frictionForce:zero(),frictionMoment:zero()});return byLink.get(name);}
    for(const m of masses){const b=bucket(m.link),f=[0,-m.mass*gravity,0];b.mass+=m.mass;b.weightedCom=add(b.weightedCom,mul(m.position,m.mass));b.gravityForce=add(b.gravityForce,f);b.gravityMoment=add(b.gravityMoment,cross(sub(m.position,com),f));}
    for(const c of loadedContacts){const b=bucket(c.link),r=sub(c.position,com);for(const [key,f]of [['support',c.forceVector],['normal',c.normalVector],['friction',c.frictionVector]]){b[key+'Force']=add(b[key+'Force'],f);b[key+'Moment']=add(b[key+'Moment'],cross(r,f));}}
    for(const joint of joints) {
      const data={mass:0,weightedCom:zero(),gravityForce:zero(),gravityMoment:zero(),supportForce:zero(),supportMoment:zero(),normalForce:zero(),normalMoment:zero(),frictionForce:zero(),frictionMoment:zero()};
      for(const name of joint.descendants){const b=byLink.get(name);if(!b)continue;data.mass+=b.mass;for(const key of ['weightedCom','gravityForce','gravityMoment','supportForce','supportMoment','normalForce','normalMoment','frictionForce','frictionMoment'])data[key]=add(data[key],b[key]);}
      const offset=sub(joint.position,com),mg=sub(data.gravityMoment,cross(offset,data.gravityForce)),mc=sub(data.supportMoment,cross(offset,data.supportForce));
      const axis=mul(joint.axis,1/(norm(joint.axis)||1)),linear=joint.type==='prismatic';
      const gravityEffort=-dot(axis,linear?data.gravityForce:mg),contactEffort=-dot(axis,linear?data.supportForce:mc),rawEffort=gravityEffort+contactEffort;
      const normalMoment=sub(data.normalMoment,cross(offset,data.normalForce)),frictionMoment=sub(data.frictionMoment,cross(offset,data.frictionForce));
      const normalEffort=-dot(axis,linear?data.normalForce:normalMoment),frictionEffort=-dot(axis,linear?data.frictionForce:frictionMoment);
      const wheelContacts=loadedContacts.filter(c=>c.wheelJoint===joint.name);
      result.joints.push({...joint,...data,subtreeCom:data.mass?mul(data.weightedCom,1/data.mass):null,axis,gravityMoment:mg,supportMoment:mc,gravityEffort,contactEffort,normalEffort,frictionEffort,isWheel:wheelContacts.length>0,wheelFriction:wheelContacts.reduce((s,c)=>s+c.frictionForce,0),wheelNormal:wheelContacts.reduce((s,c)=>s+c.normalForce,0),wheelUtilization:Math.max(0,...wheelContacts.map(c=>c.frictionUtilization)),rawEffort,effort:rawEffort,unit:linear?'N':'N·m',gravityArm:linear||!data.mass?null:gravityEffort/(data.mass*gravity),supportArm:linear||!norm(data.supportForce)?null:contactEffort/norm(data.supportForce),reactionForce:mul(add(data.gravityForce,data.supportForce),-1),reactionMoment:mul(add(mg,mc),-1)});
    }
    // URDF mimic constraints transmit virtual-work effort to the independent source.
    const rows=new Map(result.joints.map(j=>[j.name,j]));
    for(const joint of result.joints)if(joint.mimic) {
      let source=joint,multiplier=1;const seen=new Set([joint.name]);
      while(source.mimic){multiplier*=source.mimic.multiplier;source=rows.get(source.mimic.joint);if(!source||seen.has(source.name)){result.valid=false;result.reason='invalidMimic';return result;}seen.add(source.name);}
      source.effort+=multiplier*joint.rawEffort;
    }
    for(const joint of result.joints)joint.overLimit=!joint.mimic&&joint.limit!==null&&Math.abs(joint.effort)>joint.limit+1e-8;
    const force=loadedContacts.reduce((s,c)=>add(s,c.forceVector),[0,-weight,0]),moment=loadedContacts.reduce((s,c)=>add(s,cross(sub(c.position,com),c.forceVector)),zero());
    result.forceResidual=norm(force);result.momentResidual=norm(moment);
    result.totalNormal=loadedContacts.reduce((s,c)=>s+c.normalForce,0);result.totalFriction=norm(loadedContacts.reduce((s,c)=>add(s,c.frictionVector),zero()));
    result.maxFrictionUtilization=Math.max(0,...loadedContacts.map(c=>c.frictionUtilization));
    return result;
  }
  // Exact perpendicular between a joint axis and any resultant force line.
  // The shortest geometric arm differs from torque / force for inclined axes.
  function leverBreakdown(result,name) {
    const joint=result?.valid&&result.joints.find(j=>j.name===name);if(!joint)return null;
    const forces=[];
    if(joint.mass>0&&joint.subtreeCom)forces.push({id:'G',kind:'gravity',link:null,position:joint.subtreeCom,force:[0,-joint.mass*result.gravity,0]});
    const groups=new Map(),descendants=new Set(joint.descendants),inclined=Math.hypot(result.groundNormal?.[0]||0,result.groundNormal?.[2]||0)>1e-8;
    for(const c of result.contacts)if(descendants.has(c.link)&&c.force>1e-9) {
      const parts=inclined?[['normal',c.normalVector],['friction',c.frictionVector]]:[['support',c.forceVector||[0,c.force,0]]];
      for(const [kind,force]of parts){const load=norm(force);if(load<1e-9)continue;const direction=mul(force,1/load),key=[c.link,kind,...direction.map(x=>x.toFixed(9))].join('|');
        if(!groups.has(key))groups.set(key,{kind,link:c.link,load:0,weighted:zero(),force:zero()});
        const g=groups.get(key);g.load+=load;g.weighted=add(g.weighted,mul(c.position,load));g.force=add(g.force,force);
      }
    }
    const counts={support:0,normal:0,friction:0};
    for(const g of groups.values())forces.push({id:`${{support:'R',normal:'N',friction:'T'}[g.kind]}${++counts[g.kind]}`,kind:g.kind,link:g.link,position:mul(g.weighted,1/g.load),force:g.force});
    const axis=joint.axis,q=joint.position,linear=joint.type==='prismatic';
    const components=forces.map(f=>{
      const r=sub(f.position,q),magnitude=norm(f.force),direction=mul(f.force,1/magnitude),alignment=dot(axis,direction),denominator=1-alignment*alignment,effort=-dot(axis,linear?f.force:cross(r,f.force));
      if(linear||denominator<1e-8)return {...f,magnitude,effort,arm:null,effectiveArm:linear?null:effort/magnitude,parallel:!linear};
      const ar=dot(axis,r),dr=dot(direction,r),along=(ar-alignment*dr)/denominator,t=(alignment*ar-dr)/denominator;
      const axisPoint=add(q,mul(axis,along)),linePoint=add(f.position,mul(direction,t)),delta=sub(linePoint,axisPoint);
      return {...f,magnitude,effort,effectiveArm:effort/magnitude,parallel:false,arm:{axisPoint,linePoint,delta,distance:norm(delta)}};
    });
    return {name,position:q,axis,type:joint.type,unit:joint.unit,rawEffort:joint.rawEffort,actuatorEffort:joint.effort,components};
  }
  root.URDFStatics={analyze,contactLoads,frictionLoads,hull2,supportMargin,leverBreakdown};
})(typeof window==='undefined'?globalThis:window);
