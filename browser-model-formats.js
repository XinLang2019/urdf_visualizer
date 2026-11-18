/* Local model interchange for static hosting. No network or Python APIs. */
(() => {
  'use strict';
  const T=parcelRequire('../node_modules/three/build/three.module.js');
  const V=a=>new T.Vector3(...(a||[0,0,0])), I=()=>new T.Matrix4(), Q=a=>a?new T.Quaternion(a[1],a[2],a[3],a[0]).normalize():new T.Quaternion();
  const children=(e,tag)=>[...e.children].filter(n=>n.tagName===tag), child=(e,tag)=>children(e,tag)[0];
  const nums=(s,d=[])=>{const a=s==null?d:Array.isArray(s)?s:String(s).trim().split(/[\s,]+/).filter(Boolean).map(Number);if(!a.every(Number.isFinite))throw Error('Invalid numeric data / 无效数值');return a;};
  const number=(v,d=0)=>nums(v,[d])[0], arr=v=>v.toArray(), fmt=a=>a.map(v=>Number(Math.abs(v)<1e-12?0:v.toPrecision(10))).join(' ');
  const pose=(p=[0,0,0],q=new T.Quaternion())=>I().compose(V(p),q,V([1,1,1]));
  const inv=m=>m.clone().invert(), product=(a,b)=>a.clone().multiply(b), point=(m,a)=>arr(V(a).applyMatrix4(m));
  const quaternion=m=>new T.Quaternion().setFromRotationMatrix(m), xyz=m=>arr(new T.Vector3().setFromMatrixPosition(m));
  const rpy=q=>{const e=new T.Euler().setFromQuaternion(q,'ZYX');return [e.x,e.y,e.z];};
  const urdfPose=e=>{const o=child(e,'origin');return pose(nums(o?.getAttribute('xyz'),[0,0,0]),new T.Quaternion().setFromEuler(new T.Euler(...nums(o?.getAttribute('rpy'),[0,0,0]),'ZYX')));};
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const xml=text=>{const d=new DOMParser().parseFromString(text,'application/xml');if(d.querySelector('parsererror'))throw Error('Invalid XML / XML 格式错误');return d.documentElement;};
  const origin=m=>`<origin xyz="${fmt(xyz(m))}" rpy="${fmt(rpy(quaternion(m)))}"/>`;
  const normalize=p=>{if(/^[a-z]+:|^[/\\]/i.test(p))throw Error('External resource paths are unsupported / 不支持外部资源路径: '+p);const out=[];for(const s of p.replace(/\\/g,'/').split('/')){if(s==='..'){if(!out.length)throw Error('Resource outside model folder / 资源超出模型文件夹');out.pop();}else if(s&&s!=='.')out.push(s);}return out.join('/');};
  const directory=p=>p.slice(0,p.lastIndexOf('/')+1);
  function tensor(values=[0,0,0,0,0,0]){const [a,b,c,d=0,e=0,f=0]=values;return [[a,d,e],[d,b,f],[e,f,c]];}
  function rotateTensor(a,m){const e=m.elements,r=[[e[0],e[4],e[8]],[e[1],e[5],e[9]],[e[2],e[6],e[10]]];return r.map((row,i)=>r.map((_,j)=>row.reduce((s,x,k)=>s+x*r[j].reduce((v,y,l)=>v+a[k][l]*y,0),0)));}
  const tensorValues=a=>[a[0][0],a[1][1],a[2][2],a[0][1],a[0][2],a[1][2]];
  function principal(a){
    a=a.map(r=>r.slice());const u=[[1,0,0],[0,1,0],[0,0,1]];
    for(let k=0;k<40;k++){let p=0,q=1;for(const [i,j] of [[0,2],[1,2]])if(Math.abs(a[i][j])>Math.abs(a[p][q])){p=i;q=j;}if(Math.abs(a[p][q])<1e-13)break;
      const angle=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(angle),s=Math.sin(angle),b=a.map(r=>r.slice());
      for(let i=0;i<3;i++){b[i][p]=c*a[i][p]-s*a[i][q];b[i][q]=s*a[i][p]+c*a[i][q];}a=b.map(r=>r.slice());
      for(let i=0;i<3;i++){a[p][i]=c*b[p][i]-s*b[q][i];a[q][i]=s*b[p][i]+c*b[q][i];const x=u[i][p],y=u[i][q];u[i][p]=c*x-s*y;u[i][q]=s*x+c*y;}
    }
    const m=I().set(u[0][0],u[0][1],u[0][2],0,u[1][0],u[1][1],u[1][2],0,u[2][0],u[2][1],u[2][2],0,0,0,0,1);
    return {values:[a[0][0],a[1][1],a[2][2]],q:quaternion(m)};
  }
  const fresh=name=>({name:name||'robot',links:new Map(),joints:[],notes:[],note(s){if(!this.notes.includes(s))this.notes.push(s);}});
  const link=(m,name)=>{if(!name||m.links.has(name))throw Error('Duplicate / empty link name: '+name);const l={name,mass:0,com:[0,0,0],inertia:tensor(),visuals:[],collisions:[]};m.links.set(name,l);return l;};
  const joint=(name,kind,parent,child,frame=I(),axis=[1,0,0])=>({name,kind,parent,child,pose:frame,axis,lower:0,upper:0,effort:0,velocity:0,damping:0,friction:0});
  function validate(m){
    if(!m.links.size)throw Error('No model links / 未找到模型连杆');const parents=new Map(),names=new Set();
    for(const l of m.links.values())if(!Number.isFinite(l.mass)||l.mass<0||![...l.com,...l.inertia.flat()].every(Number.isFinite))throw Error('Invalid mass / inertial data / 无效质量或惯量: '+l.name);
    for(const j of m.joints)if(![...j.pose.elements,...j.axis,j.lower,j.upper,j.effort,j.velocity,j.damping,j.friction].every(Number.isFinite))throw Error('Invalid joint coordinates / 无效关节坐标: '+j.name);
    for(const j of m.joints){if(!j.name||names.has(j.name)||!m.links.has(j.parent)||!m.links.has(j.child)||parents.has(j.child)||j.parent===j.child)throw Error('Invalid / closed-loop joint: '+j.name);names.add(j.name);parents.set(j.child,j.parent);if(!['fixed','floating','revolute','continuous','prismatic'].includes(j.kind))throw Error('Unsupported joint type: '+j.kind);if(j.lower>j.upper)throw Error('Invalid joint range: '+j.name);if(!['fixed','floating'].includes(j.kind)){const v=V(j.axis);if(v.length()<1e-10)throw Error('Zero joint axis: '+j.name);j.axis=arr(v.normalize());}}
    for(const name of m.links.keys()){const seen=new Set();let n=name;while(parents.has(n)){if(seen.has(n))throw Error('Closed-loop model / 模型存在闭环');seen.add(n);n=parents.get(n);}}
    const roots=[...m.links.keys()].filter(n=>!parents.has(n));if(roots.length!==1)throw Error('Expected one connected robot tree / 模型需要一棵连通树');return roots[0];
  }
  function worlds(m){const root=validate(m),w=new Map([[root,I()]]);while(w.size<m.links.size)for(const j of m.joints)if(w.has(j.parent)&&!w.has(j.child))w.set(j.child,product(w.get(j.parent),j.pose));return w;}
  class Resources {
    constructor(entry,files,options={}){this.entry=normalize(entry);this.files=files;this.cache=new Map();this.options=options;}
    resolve(ref,base=this.entry){let p=ref.replace(/^package:\/\//,'');const candidates=[];try{candidates.push(normalize(directory(base)+p));}catch{}try{candidates.push(normalize(p));}catch{}
      for(const key of candidates)if(this.files.has(key))return key;
      let matches=[...this.files.keys()].filter(k=>k.endsWith('/'+p)||p.endsWith('/'+k));if(matches.length===1)return matches[0];
      const file=p.split('/').pop();matches=[...this.files.keys()].filter(k=>k.split('/').pop()===file);if(matches.length===1)return matches[0];
      throw Error('Missing or ambiguous resource / 资源缺失或名称不唯一: '+ref);
    }
    text(path=this.entry){this.options.signal?.throwIfAborted();return this.files.get(path).text();}
    async mesh(ref,scale=[1,1,1],base=this.entry){const path=this.resolve(ref,base),key=path+'|'+scale.join();if(!this.cache.has(key))this.cache.set(key,this.loadMesh(path,scale));return this.cache.get(key);}
    async loadMesh(path,scale){
      const file=this.files.get(path),ext=path.split('.').pop().toLowerCase(),manager=new T.LoadingManager();manager.setURLModifier(()=>{throw Error('Collect external mesh resources in the model folder / 请收集网格依赖文件');});let scene;
      this.options.signal?.throwIfAborted();
      if(this.options.step&&['dae','obj'].includes(ext)&&file.size>32*1024*1024)throw Error('STEP: text mesh exceeds 32 MiB; simplify it first. / 文本网格超过 32 MiB，请先简化: '+path);
      if(this.options.step&&ext==='stl'){
        const header=await file.slice(0,84).arrayBuffer(),count=header.byteLength===84?new DataView(header).getUint32(80,true):0,binary=count>0&&84+count*50<=file.size;
        if(binary&&count>250000)throw Error('STEP: a mesh exceeds 250,000 triangles; simplify it with Mesh Simplifier first. / 单个网格超过 25 万面，请先用 Mesh Simplifier 简化: '+path);
        if(!binary&&file.size>32*1024*1024)throw Error('STEP: text STL exceeds 32 MiB; simplify it with Mesh Simplifier first. / 文本 STL 超过 32 MiB，请先用 Mesh Simplifier 简化: '+path);
      }
      if(ext==='stl'){const Loader=parcelRequire('../node_modules/three/examples/jsm/loaders/STLLoader.js').STLLoader;scene=new T.Mesh(new Loader(manager).parse(await file.arrayBuffer()));}
      else if(ext==='obj'){const Loader=parcelRequire('../node_modules/three/examples/jsm/loaders/OBJLoader.js').OBJLoader;scene=new Loader(manager).parse(await file.text());}
      else if(ext==='dae'){const Loader=parcelRequire('../node_modules/three/examples/jsm/loaders/ColladaLoader.js').ColladaLoader;scene=new Loader(manager).parse(await file.text(),'').scene;scene.quaternion.identity();}
      else throw Error('Browser mesh conversion supports STL, OBJ and DAE / 浏览器转换支持 STL、OBJ、DAE 网格: '+path);
      if(this.options.step){
        this.options.signal?.throwIfAborted();scene.updateMatrixWorld(true);let pointCount=0,faceCount=0;
        scene.traverse(o=>{const p=o.geometry?.attributes.position;if(p){pointCount+=p.count;faceCount+=Math.floor((o.geometry.index?.count||p.count)/3);}});
        if(faceCount>250000)throw Error('STEP: a mesh exceeds 250,000 triangles; simplify it with Mesh Simplifier first. / 单个网格超过 25 万面，请先用 Mesh Simplifier 简化: '+path);
        if(!pointCount||!faceCount)throw Error('Empty mesh / 空网格: '+path);
        const points=new Float64Array(pointCount*3),faces=new Uint32Array(faceCount*3),scaling=I().makeScale(...scale),p=new T.Vector3();let offset=0,f=0;
        scene.traverse(o=>{const a=o.geometry?.attributes.position;if(!a)return;const matrix=product(scaling,o.matrixWorld);
          for(let i=0;i<a.count;i++){p.fromBufferAttribute(a,i).applyMatrix4(matrix);points.set([p.x,p.y,p.z],(offset+i)*3);}
          const index=o.geometry.index,count=index?index.count:a.count,reverse=matrix.determinant()<0;
          for(let i=0;i+2<count;i+=3)for(const k of reverse?[2,1,0]:[0,1,2])faces[f++]=offset+(index?index.getX(i+k):i+k);
          offset+=a.count;o.geometry.dispose();
        });
        this.options.onProgress?.({stage:'reading',name:path,progress:.05});await new Promise(resolve=>setTimeout(resolve,0));this.options.signal?.throwIfAborted();
        return {points,faces};
      }
      scene.updateMatrixWorld(true);const points=[],faces=[],scaling=I().makeScale(...scale);
      scene.traverse(o=>{const p=o.geometry?.attributes?.position;if(!p)return;const matrix=product(scaling,o.matrixWorld),offset=points.length;for(let i=0;i<p.count;i++)points.push(point(matrix,[p.getX(i),p.getY(i),p.getZ(i)]));const idx=o.geometry.index,count=idx?idx.count:p.count;for(let i=0;i+2<count;i+=3){const f=[0,1,2].map(k=>offset+(idx?idx.getX(i+k):i+k));if(matrix.determinant()<0)f.reverse();faces.push(f);}});
      if(!points.length||!faces.length)throw Error('Empty mesh / 空网格: '+path);return {points,faces};
    }
  }
  const meshFromGeometry=g=>{const p=g.attributes.position,points=[],faces=[];for(let i=0;i<p.count;i++)points.push([p.getX(i),p.getY(i),p.getZ(i)]);const idx=g.index,count=idx?idx.count:p.count;for(let i=0;i<count;i+=3)faces.push([0,1,2].map(k=>idx?idx.getX(i+k):i+k));g.dispose();return {points,faces};};
  function primitive(kind,size){
    if(size.some(v=>!(v>0)))throw Error('Geometry dimensions must be positive / 几何尺寸必须大于零');let g;
    if(kind==='box')g=new T.BoxGeometry(...size);
    else if(kind==='sphere')g=new T.SphereGeometry(size[0],24,16);
    else if(kind==='ellipsoid'){g=new T.SphereGeometry(1,24,16);g.scale(...size);}
    else if(kind==='cylinder'||kind==='cone'){g=new T.CylinderGeometry(kind==='cone'?0:size[0],size[0],size[1],24);g.rotateX(Math.PI/2);}
    else if(kind==='capsule'){
      const [r,h]=size,points=[],faces=[],segments=24,rings=[];
      for(const side of [-1,1])for(let k=0;k<=8;k++){const phi=(side<0?-Math.PI/2:0)+k*Math.PI/16;rings.push([r*Math.cos(phi),side*h/2+r*Math.sin(phi)]);}
      for(const [radius,z] of rings)for(let k=0;k<segments;k++){const theta=k*2*Math.PI/segments;points.push([radius*Math.cos(theta),radius*Math.sin(theta),z]);}
      for(let row=0;row+1<rings.length;row++)for(let k=0;k<segments;k++){const a=row*segments+k,b=row*segments+(k+1)%segments,c=b+segments,d=a+segments;if(rings[row][0]>1e-12)faces.push([a,b,c]);if(rings[row+1][0]>1e-12)faces.push([a,c,d]);}return {points,faces};
    }
    else throw Error('Unsupported geometry / 不支持的几何: '+kind);return meshFromGeometry(g);
  }
  const shape=(kind,size,frame=I(),color=[.72,.78,.84,1])=>({kind,size,pose:frame,color});
  const geometryMesh=s=>s.mesh||primitive(s.kind,s.size);
  function shiftedMass(l,frame){l.com=point(frame,l.com);l.inertia=rotateTensor(l.inertia,frame);for(const s of new Set([...l.visuals,...l.collisions]))s.pose=product(frame,s.pose);}
  async function readURDF(r){
    const root=xml(await r.text());if(root.tagName!=='robot')throw Error('Expected <robot>');const m=fresh(root.getAttribute('name')),materials=new Map(children(root,'material').map(e=>[e.getAttribute('name'),e]));
    for(const e of children(root,'link')){
      const l=link(m,e.getAttribute('name')),a=child(e,'inertial');if(a){const mass=child(a,'mass'),i=child(a,'inertia');if(!mass||!i)throw Error('Incomplete inertial / 惯量信息不完整: '+l.name);l.mass=number(mass.getAttribute('value'));const frame=urdfPose(a);l.com=xyz(frame);l.inertia=rotateTensor(tensor(['ixx','iyy','izz','ixy','ixz','iyz'].map(k=>number(i.getAttribute(k)))),frame);}
      for(const [tag,key] of [['visual','visuals'],['collision','collisions']])for(const g of children(e,tag)){
        if(r.options.step&&tag==='collision'&&children(e,'visual').length)continue;
        const geo=child(g,'geometry')?.firstElementChild;if(!geo)throw Error('Missing geometry: '+l.name);let mat=child(g,'material');if(mat&&!child(mat,'color'))mat=materials.get(mat.getAttribute('name'))||mat;
        const color=nums(child(mat||g,'color')?.getAttribute('rgba'),[.72,.78,.84,1]),s=shape(geo.tagName,[],urdfPose(g),color);
        if(s.kind==='mesh')s.mesh=await r.mesh(geo.getAttribute('filename'),nums(geo.getAttribute('scale'),[1,1,1]));
        else if(s.kind==='box')s.size=nums(geo.getAttribute('size'));
        else if(['sphere','cylinder'].includes(s.kind))s.size=[number(geo.getAttribute('radius')),...(s.kind==='cylinder'?[number(geo.getAttribute('length'))]:[])];
        else throw Error('Unsupported URDF geometry: '+s.kind);
        l[key].push(s);
      }
    }
    for(const e of children(root,'joint')){const j=joint(e.getAttribute('name'),e.getAttribute('type'),child(e,'parent')?.getAttribute('link'),child(e,'child')?.getAttribute('link'),urdfPose(e),nums(child(e,'axis')?.getAttribute('xyz'),[1,0,0]));const lim=child(e,'limit'),dyn=child(e,'dynamics'),mi=child(e,'mimic');for(const k of ['lower','upper','effort','velocity'])j[k]=number(lim?.getAttribute(k));j.damping=number(dyn?.getAttribute('damping'));j.friction=number(dyn?.getAttribute('friction'));if(mi)j.mimic={joint:mi.getAttribute('joint'),multiplier:number(mi.getAttribute('multiplier'),1),offset:number(mi.getAttribute('offset'))};m.joints.push(j);}
    if(root.querySelector('mesh,texture'))m.note('Meshes use a constant RGBA color; textures and per-face materials are not transferred. / 网格保留常量颜色，不迁移贴图和面材质。');
    if(root.querySelector('transmission,gazebo,ros2_control'))m.note('Simulator and controller extensions are not transferred. / 不迁移仿真器与控制器扩展。');validate(m);return m;
  }
  async function expandMJCF(r,path=r.entry,stack=[]){
    if(stack.includes(path)||stack.length>32)throw Error('Cyclic / deep MJCF includes');const root=xml(await r.text(path));for(const e of [...root.querySelectorAll('include')]){const p=r.resolve(e.getAttribute('file'),path),part=await expandMJCF(r,p,[...stack,path]);e.replaceWith(...[...part.children].map(n=>root.ownerDocument.importNode(n,true)));}return root;
  }
  async function readMJCF(r){
    const root=await expandMJCF(r),m=fresh(root.getAttribute('model'));if(root.tagName!=='mujoco')throw Error('Expected <mujoco>');
    if(root.querySelector('replicate,composite,flex,flexcomp,attach,plugin'))throw Error('MJCF procedural / plugin models must be expanded before browser import / 请先展开 MJCF 过程式模型和插件。');
    const compiler=child(root,'compiler'),get=(e,k,d)=>e?.getAttribute(k)??d,angle=get(compiler,'angle','degree')==='radian'?1:Math.PI/180,seq=get(compiler,'eulerseq','xyz');
    if(get(compiler,'coordinate','local')!=='local')throw Error('MJCF global coordinates require conversion to local coordinates');
    if(number(get(compiler,'settotalmass',null))>0)throw Error('MJCF settotalmass must be baked into body inertials before browser import / 请先将 MJCF 总质量缩放烘焙到各刚体。');
    const meshdir=get(compiler,'meshdir',get(compiler,'assetdir','')),inertiaGroups=nums(get(compiler,'inertiagrouprange',null),[0,5]),defaults=new Map();
    function defaultsAt(e,inherited=new Map()){
      const values=new Map([...inherited].map(([k,v])=>[k,{...v}]));for(const n of [...e.children].filter(n=>n.tagName!=='default'))values.set(n.tagName,{...values.get(n.tagName),...Object.fromEntries([...n.attributes].map(a=>[a.name,a.value]))});
      defaults.set(e.getAttribute('class')||'',values);for(const n of children(e,'default'))defaultsAt(n,values);
    }
    for(const e of children(root,'default'))defaultsAt(e);
    function attrs(e,cls=''){return {...defaults.get(e.getAttribute('class')||cls||'')?.get(e.tagName),...Object.fromEntries([...e.attributes].map(a=>[a.name,a.value]))};}
    function rotation(a){
      if(a.quat)return Q(nums(a.quat));if(a.axisangle){const v=nums(a.axisangle);return new T.Quaternion().setFromAxisAngle(V(v.slice(0,3)).normalize(),v[3]*angle);}
      if(a.euler){const v=nums(a.euler),q=new T.Quaternion();for(let i=0;i<3;i++){const n=seq[i],axis=[0,0,0];axis['xyz'.indexOf(n.toLowerCase())]=1;const d=new T.Quaternion().setFromAxisAngle(V(axis),v[i]*angle);if(n===n.toLowerCase())q.multiply(d);else q.premultiply(d);}return q;}
      if(a.xyaxes){const v=nums(a.xyaxes),x=V(v.slice(0,3)).normalize(),y=V(v.slice(3)).normalize(),z=x.clone().cross(y).normalize();y.copy(z).cross(x);return quaternion(I().makeBasis(x,y,z));}
      if(a.zaxis)return new T.Quaternion().setFromUnitVectors(V([0,0,1]),V(nums(a.zaxis)).normalize());return new T.Quaternion();
    }
    const frame=a=>pose(nums(a.pos,[0,0,0]),rotation(a)),meshes=new Map(),materials=new Map();
    for(const e of root.querySelectorAll('asset > mesh'))meshes.set(e.getAttribute('name')||e.getAttribute('file')?.split('/').pop().replace(/\.[^.]+$/,''),attrs(e));
    for(const e of root.querySelectorAll('asset > material'))materials.set(e.getAttribute('name'),attrs(e));
    let serial=0;const worldName=()=>{let name='__world';while(m.links.has(name)||[...root.querySelectorAll('body')].some(e=>e.getAttribute('name')===name))name+='_';return name;};
    const metadata=(()=>{try{return JSON.parse(root.querySelector('custom > text[name="urdf_viewer_metadata"]')?.getAttribute('data')||'{}');}catch{return {};}})(),world=link(m,metadata.worldRoot||worldName());
    async function body(e,parent,offset=I(),cls=''){
      const a=attrs(e,cls),name=a.name||'body_'+(++serial),l=link(m,name),bodyPose=product(offset,frame(a)),rawJoints=[...e.children].filter(n=>['joint','freejoint'].includes(n.tagName));
      if(rawJoints.length>1)throw Error('Browser import currently requires at most one joint per MJCF body / 单个 MJCF body 暂支持一个关节，请先展开串联关节: '+name);
      const je=rawJoints[0],ja=je?attrs(je,a.childclass||cls):{},type=je?.tagName==='freejoint'?'free':ja.type||'hinge';
      if(je&&type==='ball')throw Error('Ball joints are not supported by the URDF workbench / 暂不支持球关节: '+name);
      const anchor=je&&type!=='free'?nums(ja.pos,[0,0,0]):[0,0,0],rebase=pose(anchor.map(x=>-x));
      const j=joint(ja.name||metadata.fixedJointNames?.[name]||name+'_joint',!je?'fixed':type==='free'?'floating':type==='slide'?'prismatic':'continuous',parent,name,product(bodyPose,pose(anchor)),nums(ja.axis,[0,0,1]));
      const range=nums(ja.range),limited=ja.limited==='true'||(ja.limited!=='false'&&range.length===2);if(je&&type!=='free'){
        const unit=type==='slide'?1:angle,ref=number(ja.ref)*unit;
        if(limited){j.kind=type==='slide'?'prismatic':'revolute';j.lower=range[0]*unit-ref;j.upper=range[1]*unit-ref;}else if(type==='slide'){j.lower=-1e6;j.upper=1e6;m.note('Unbounded slide joints use ±1000000 m viewer limits. / 无限位移动关节使用 ±1000000 m 显示范围。');}
        j.damping=number(ja.damping);j.friction=number(ja.frictionloss);const effort=nums(ja.actuatorfrcrange);if(effort.length)j.effort=Math.max(...effort.map(Math.abs));
        const extra=metadata.joints?.[j.name];if(extra){j.velocity=extra.velocity||0;j.effort=extra.effort||j.effort;j.mimic=extra.mimic;}
      }m.joints.push(j);
      const inertial=get(compiler,'inertiafromgeom','auto')==='true'?null:child(e,'inertial');if(inertial){const i=attrs(inertial),f=frame(i);l.mass=number(i.mass);l.com=xyz(f);l.inertia=rotateTensor(tensor(nums(i.fullinertia||i.diaginertia,[0,0,0])),f);}
      for(const ge of children(e,'geom')){
        const g=attrs(ge,a.childclass||cls),kind=g.type||(g.mesh?'mesh':'sphere');if(kind==='plane'){m.note('Infinite ground planes are omitted. / 忽略无限地面。');continue;}
        const size=nums(g.size),color=nums(g.rgba||materials.get(g.material)?.rgba,[.72,.78,.84,1]);let p=frame(g),s;
        if(g.fromto){const v=nums(g.fromto),start=V(v.slice(0,3)),end=V(v.slice(3)),delta=end.clone().sub(start);if(delta.length()<1e-10)throw Error('Zero length geom fromto');p=pose(arr(start.add(end).multiplyScalar(.5)),new T.Quaternion().setFromUnitVectors(V([0,0,1]),delta.clone().normalize()));size[1]=delta.length()/2;}
        if(kind==='mesh'){
          const def=meshes.get(g.mesh);if(!def)throw Error('Missing MJCF mesh: '+g.mesh);let data;
          if(def.file)data=await r.mesh(meshdir?meshdir.replace(/\/$/,'')+'/'+def.file:def.file,[1,1,1]);
          else{const v=nums(def.vertex),f=nums(def.face);if(!v.length||!f.length)throw Error('Inline mesh needs vertex and face arrays');data={points:Array.from({length:v.length/3},(_,i)=>v.slice(i*3,i*3+3)),faces:Array.from({length:f.length/3},(_,i)=>f.slice(i*3,i*3+3))};}
          const scale=nums(def.scale,[1,1,1]),ref=V(nums(def.refpos,[0,0,0])),q=Q(nums(def.refquat,[1,0,0,0])).invert();s=shape('mesh',[],p,color);s.mesh={points:data.points.map(v=>arr(V(v).sub(ref).applyQuaternion(q).multiply(V(scale)))),faces:data.faces.map(f=>scale.reduce((x,y)=>x*y,1)<0?f.slice().reverse():f)};
        }else{s=shape(kind,kind==='box'?size.map(x=>x*2):['cylinder','capsule'].includes(kind)?[size[0],2*size[1]]:kind==='ellipsoid'?size:[size[0]],p,color);primitive(s.kind,s.size);}
        const collision=number(g.contype,1)!==0||number(g.conaffinity,1)!==0;if(collision)l.collisions.push(s);if(number(g.group)!==3||!collision)l.visuals.push(s);
        if(!inertial&&get(compiler,'inertiafromgeom','auto')!=='false'&&number(g.group)>=inertiaGroups[0]&&number(g.group)<=inertiaGroups[1]){
          if(kind==='mesh')m.note('Mesh-only bodies without explicit inertia retain unknown mass. / 纯网格刚体缺少显式惯量时保留未知质量。');
          else addPrimitiveMass(l,s,g.mass==null?null:number(g.mass),number(g.density,1000));
        }
      }
      shiftedMass(l,rebase);
      for(const c of children(e,'body'))await body(c,name,rebase,a.childclass||cls);
    }
    const wb=child(root,'worldbody');if(!wb)throw Error('MJCF worldbody is missing');
    if(children(wb,'geom').some(e=>(e.getAttribute('type')||'sphere')!=='plane'))throw Error('Move MJCF world geometry into named fixed bodies before browser import / 请先把 MJCF 世界几何放入命名的固定 body。');
    if(children(wb,'geom').length)m.note('Infinite ground planes are omitted. / 忽略无限地面。');
    for(const b of children(wb,'body'))await body(b,world.name);
    for(const motor of root.querySelectorAll('actuator > *')){const a=attrs(motor),j=m.joints.find(j=>j.name===a.joint);if(!j)continue;const range=nums(a.forcerange||(motor.tagName==='motor'?a.ctrlrange:null));if(range.length&&a.forcelimited!=='false')j.effort=Math.max(...range.map(Math.abs))*Math.abs(nums(a.gear,[1])[0]);}
    if(root.querySelector('sensor,tendon,equality,actuator'))m.note('Controllers, sensors, tendons and solver constraints are not transferred as simulation equivalents. / 控制器、传感器、肌腱与求解约束不作仿真等价迁移。');
    if(!world.visuals.length&&!world.collisions.length){const roots=m.joints.filter(j=>j.parent===world.name);if(roots.length===1&&roots[0].kind==='fixed'&&roots[0].pose.elements.every((v,i)=>Math.abs(v-I().elements[i])<1e-10)){m.joints.splice(m.joints.indexOf(roots[0]),1);m.links.delete(world.name);}}
    validate(m);return m;
  }
  function addPrimitiveMass(l,s,explicit,density){
    const d=s.size;let volume,diag;
    if(s.kind==='box'){volume=d[0]*d[1]*d[2];diag=[d[1]**2+d[2]**2,d[0]**2+d[2]**2,d[0]**2+d[1]**2].map(x=>x/12);}
    else if(s.kind==='sphere'){volume=4*Math.PI*d[0]**3/3;diag=[1,1,1].map(()=>.4*d[0]**2);}
    else if(s.kind==='ellipsoid'){volume=4*Math.PI*d[0]*d[1]*d[2]/3;diag=[d[1]**2+d[2]**2,d[0]**2+d[2]**2,d[0]**2+d[1]**2].map(x=>x/5);}
    else if(s.kind==='cylinder'){volume=Math.PI*d[0]**2*d[1];diag=[(3*d[0]**2+d[1]**2)/12,(3*d[0]**2+d[1]**2)/12,d[0]**2/2];}
    else if(s.kind==='capsule'){const [r,h]=d,vc=Math.PI*r*r*h,vs=4*Math.PI*r**3/3;volume=vc+vs;const transverse=(vc*(3*r*r+h*h)/12+vs*(.4*r*r+h*h/4+3*h*r/8))/volume;diag=[transverse,transverse,(vc*r*r/2+vs*.4*r*r)/volume];}
    else return;
    const mass=explicit??density*volume;if(!(mass>0))return;const com=xyz(s.pose),old=l.mass,total=mass+old,newCom=l.com.map((v,i)=>(v*old+com[i]*mass)/total),own=rotateTensor(tensor(diag.map(v=>v*mass)),s.pose),a=l.com.map((v,i)=>v-newCom[i]),b=com.map((v,i)=>v-newCom[i]);
    l.inertia=l.inertia.map((row,i)=>row.map((v,j)=>v+own[i][j]+old*((i===j?a.reduce((s,x)=>s+x*x,0):0)-a[i]*a[j])+mass*((i===j?b.reduce((s,x)=>s+x*x,0):0)-b[i]*b[j])));l.com=newCom;l.mass=total;
  }
  // Tokenize USDA rather than splitting lines: arrays, tuples and metadata can span lines.
  function parseUSDA(text){
    if(!text.trimStart().startsWith('#usda'))throw Error('Browser mode supports text USDA only; binary USD / USDZ needs the optional native service. / 浏览器模式支持文本 USDA，二进制 USD / USDZ 请先转换为 .usda 或使用本地原生服务。');
    const pattern=/\s+|#[^\n]*|"(?:\\.|[^"\\])*"|@[^@]*@|<[^>]*>|[{}()[\]=,;]|[^\s{}()[\]=,;]+/gy,tokens=[];let pos=0,match;
    while(pos<text.length){pattern.lastIndex=pos;match=pattern.exec(text);if(!match)throw Error('Invalid USDA syntax at '+pos);pos=pattern.lastIndex;const s=match[0];if(!/^\s|^#/.test(s))tokens.push(s);}
    let i=0,depth=0;const peek=()=>tokens[i],take=()=>tokens[i++],expect=s=>{if(take()!==s)throw Error('USDA expected '+s+' near token '+i);};
    function value(){const s=take();if(s===undefined)throw Error('Incomplete USDA value');if(['[','('].includes(s)){if(++depth>100)throw Error('USDA nesting too deep');const a=[],end=s==='['?']':')';while(peek()!==end){if(!peek())throw Error('Unclosed USDA array');if([',',';'].includes(peek())){take();continue;}a.push(value());}take();depth--;return a;}
      if(s==='{')throw Error('USDA dictionaries / timeSamples require flattening at a default time. / 请先将 USDA 字典和动画烘焙为静态数据。');
      if(s.startsWith('"'))return JSON.parse(s);if(s.startsWith('<'))return {path:s.slice(1,-1)};if(s.startsWith('@'))return {asset:s.slice(1,-1)};if(s==='true')return true;if(s==='false')return false;if(/^[+-]?(?:\d|\.\d)/.test(s)||/^[+-]?(inf|nan)$/i.test(s))return Number(s);return s;
    }
    function property(out){
      const header=[];while(peek()&&peek()!=='='&&!['}','def','over','class'].includes(peek())){if(peek()==='('){metadata();break;}header.push(take());if(header.length>16)throw Error('Unsupported USDA property syntax');}
      if(peek()!=='=')throw Error('USDA property needs a value near '+header.join(' '));take();const key=header[header.length-1];if(['references','payload','subLayers','inherits','specializes','variantSets','variants'].includes(key))throw Error('USD composition / variants require a flattened .usda file. / 请先把 USD 引用层、变体与载荷展开为单个 .usda。');
      if(key.includes('.timeSamples'))throw Error('USDA animation must be baked at the desired pose / 请先将 USD 动画烘焙到所需姿态');out[key]=value();if(peek()==='(')metadata();if([',',';'].includes(peek()))take();
    }
    function metadata(){expect('(');const a={};while(peek()!==')'){if(!peek())throw Error('Unclosed USDA metadata');property(a);}take();return a;}
    function prim(parent){
      if(++depth>100)throw Error('USDA prim nesting too deep');const spec=take();let type='',name=take();if(!name?.startsWith('"')){type=name;name=take();}if(!name?.startsWith('"'))throw Error('Invalid USDA prim name');name=JSON.parse(name);
      if(spec!=='def')throw Error('USDA over / class specs require flattening / 请先展开 USDA over / class 定义');
      const node={name,type,path:(parent?.path||'')+'/'+name,parent,props:{},meta:peek()==='('?metadata():{},children:[]};expect('{');
      while(peek()!=='}'){if(!peek())throw Error('Unclosed USDA prim');if(['def','over','class'].includes(peek()))node.children.push(prim(node));else property(node.props);}take();depth--;return node;
    }
    const stage={meta:peek()==='('?metadata():{},children:[]};while(peek())stage.children.push(prim(null));return stage;
  }
  const flatten=a=>Array.isArray(a)?a.flat(Infinity):[a];
  function usdLocal(node){
    const p=node.props,order=p.xformOpOrder||[],out=I();for(let key of order){if(key==='!resetXformStack!')continue;const inverse=key.startsWith('!invert!');if(inverse)key=key.slice(8);const v=flatten(p[key]??[]),type=key.split(':')[1];let op;
      if(type==='translate')op=pose(nums(v));else if(type==='scale')op=I().makeScale(...nums(v));else if(type==='orient')op=pose([0,0,0],Q(nums(v)));
      else if(type==='transform'){if(v.length!==16)throw Error('Invalid USD matrix');op=I().fromArray(nums(v));}
      else if(/^rotate[XYZ]{1,3}$/.test(type)){const axes=type.slice(6),q=new T.Quaternion();for(const axis of axes){const unit=[0,0,0],k='XYZ'.indexOf(axis);unit[k]=1;q.premultiply(new T.Quaternion().setFromAxisAngle(V(unit),(axes.length===1?v[0]:v[k])*Math.PI/180));}op=pose([0,0,0],q);}
      else throw Error('Unsupported USD transform: '+key);out.multiply(inverse?inv(op):op);
    }return out;
  }
  async function readUSD(r){
    const file=r.files.get(r.entry),header=new TextDecoder().decode(await file.slice(0,16).arrayBuffer());if(!header.trimStart().startsWith('#usda'))throw Error('Browser mode supports text USDA; binary USD / USDZ is unsupported. / 请加载文本 .usda；二进制 USD / USDZ 可先转换为 .usda，或使用本地原生服务。');
    const stage=parseUSDA(await file.text()),all=[],byPath=new Map(),unit=number(stage.meta.metersPerUnit,.01),massUnit=number(stage.meta.kilogramsPerUnit,1),up=stage.meta.upAxis||'Y';if(!(unit>0)||!(massUnit>0)||!['Y','Z'].includes(up))throw Error('Invalid USD units / up axis');
    const upQ=up==='Y'?new T.Quaternion().setFromAxisAngle(V([1,0,0]),Math.PI/2):new T.Quaternion(),conversion=I().compose(V(),upQ,V([unit,unit,unit]));
    function walk(n,parent=I()){n.world=(n.props.xformOpOrder||[]).includes('!resetXformStack!')?usdLocal(n):product(parent,usdLocal(n));all.push(n);byPath.set(n.path,n);for(const c of n.children)walk(c,n.world);}for(const n of stage.children)walk(n);
    const modelRoot=all.find(n=>n.props['urdf:modelName']),m=fresh(modelRoot?.props['urdf:modelName']||stage.meta.defaultPrim||file.name.replace(/\.[^.]+$/,'')),bodies=all.filter(n=>(n.meta.apiSchemas||[]).includes('PhysicsRigidBodyAPI')),jointNodes=all.filter(n=>/^Physics.*Joint$/.test(n.type));
    const newFrames=new Map(),bodyLink=new Map(),oldWorld=new Map();
    function rigidFrame(n){const matrix=product(conversion,n.world),p=new T.Vector3(),q=new T.Quaternion(),s=new T.Vector3();matrix.decompose(p,q,s);if([s.x,s.y,s.z].some(x=>Math.abs(x-unit)>unit*1e-5)||matrix.determinant()<=0)throw Error('USD rigid body scaling / reflection must be baked / 请先烘焙 USD 刚体缩放: '+n.path);return pose(arr(p),q);}
    if(!bodies.length){link(m,'world');for(const n of all)n.owner='world';m.note('USD contains no rigid body APIs; imported as one static geometry link. / USD 无刚体定义，按单个静态几何连杆导入。');}
    for(const n of bodies){const l=link(m,n.props['urdf:name']||n.name);bodyLink.set(n.path,l);oldWorld.set(n.path,rigidFrame(n));newFrames.set(n.path,oldWorld.get(n.path));const p=n.props;l.mass=number(p['physics:mass'])*massUnit;l.com=nums(p['physics:centerOfMass'],[0,0,0]).map(x=>x*unit);const diag=nums(p['physics:diagonalInertia'],[0,0,0]).map(x=>x*massUnit*unit*unit);l.inertia=rotateTensor(tensor(diag),pose([0,0,0],Q(nums(p['physics:principalAxes'],[1,0,0,0]))));if(!p['physics:mass'])m.note('Missing USD masses are left unknown; no collision-density mass is invented. / USD 缺失质量时保留未知值，不推测碰撞体质量。');if(l.mass>0&&!p['physics:diagonalInertia'])m.note('USD mass has no explicit inertia; MJCF export requires valid inertial data. / USD 有质量但缺少显式惯量，导出 MJCF 前需补全。');}
    const target=(n,key)=>{const value=n.props[key],path=(Array.isArray(value)?value[0]:value)?.path;if(!path||path.startsWith('/'))return path;const parts=n.path.split('/').filter(Boolean);for(const s of path.split('/')){if(s==='..')parts.pop();else if(s&&s!=='.')parts.push(s);}return '/'+parts.join('/');},incoming=new Map();
    for(const n of jointNodes){if(n.props['physics:jointEnabled']===false)continue;if(!['PhysicsFixedJoint','PhysicsRevoluteJoint','PhysicsPrismaticJoint'].includes(n.type))throw Error('Unsupported USD joint / 暂不支持的 USD 关节: '+n.type);const b=target(n,'physics:body1');if(!bodyLink.has(b))throw Error('USD joint body1 is missing: '+n.path);if(incoming.has(b))throw Error('USD multi-parent / closed-loop constraints need flattening');incoming.set(b,n);const f=pose(nums(n.props['physics:localPos1'],[0,0,0]).map(x=>x*unit),Q(nums(n.props['physics:localRot1'],[1,0,0,0])));newFrames.set(b,product(oldWorld.get(b),f));}
    let worldName='__world';while(m.links.has(worldName))worldName+='_';if(bodies.length)link(m,worldName);
    for(const n of bodies){const l=bodyLink.get(n.path),jnode=incoming.get(n.path),p=jnode?.props||{},parentPath=jnode&&target(jnode,'physics:body0');if(parentPath&&!bodyLink.has(parentPath))throw Error('USD joint body0 is missing: '+parentPath);
      const parent=parentPath?bodyLink.get(parentPath).name:worldName,parentWorld=parentPath?newFrames.get(parentPath):I(),kind=!jnode?(n.props['urdf:floatingJointName']?'floating':'fixed'):jnode.type==='PhysicsFixedJoint'?'fixed':jnode.type==='PhysicsPrismaticJoint'?'prismatic':p['physics:lowerLimit']==null&&p['physics:upperLimit']==null?'continuous':'revolute';
      const frame0=jnode?pose(nums(p['physics:localPos0'],[0,0,0]).map(x=>x*unit),Q(nums(p['physics:localRot0'],[1,0,0,0]))):newFrames.get(n.path),jointWorld=jnode?product(parentPath?oldWorld.get(parentPath):I(),frame0):newFrames.get(n.path);
      if(jnode&&V(xyz(jointWorld)).distanceTo(V(xyz(newFrames.get(n.path))))>1e-5)throw Error('USD joint frames do not coincide at the authored pose / USD 两侧关节原点不重合，请先烘焙零位: '+jnode.path);
      if(jnode&&1-Math.abs(quaternion(jointWorld).dot(quaternion(newFrames.get(n.path))))>1e-6)throw Error('USD joint frames have a nonzero authored rotation; bake the joint zero pose first / USD 关节含非零初始转角，请先烘焙零位。');
      const axis=[0,0,0],axisIndex='XYZ'.indexOf(p['physics:axis']||'X');if(axisIndex<0)throw Error('Invalid USD joint axis');axis[axisIndex]=1;const j=joint(p['urdf:name']||n.props['urdf:floatingJointName']||jnode?.name||l.name+'_joint',kind,parent,l.name,product(inv(parentWorld),jointWorld),axis),factor=kind==='prismatic'?unit:Math.PI/180;
      j.lower=number(p['physics:lowerLimit'])*factor;j.upper=number(p['physics:upperLimit'])*factor;j.effort=number(p['urdf:effort']??p['drive:angular:physics:maxForce']??p['drive:linear:physics:maxForce']);j.velocity=number(p['urdf:velocity']);j.damping=number(p['urdf:damping']);j.friction=number(p['urdf:friction']);if(p['urdf:mimic'])j.mimic=JSON.parse(p['urdf:mimic']);m.joints.push(j);
      shiftedMass(l,product(inv(newFrames.get(n.path)),oldWorld.get(n.path)));
    }
    const geometryTypes=['Mesh','Cube','Sphere','Cylinder','Capsule','Cone'];
    for(const n of all.filter(n=>geometryTypes.includes(n.type))){
      let ancestor=n;while(ancestor&&!bodyLink.has(ancestor.path))ancestor=ancestor.parent;const l=ancestor?bodyLink.get(ancestor.path):bodies.length?m.links.get(worldName):m.links.get('world'),base=ancestor?newFrames.get(ancestor.path):I(),relative=product(inv(base),product(conversion,n.world)),p=n.props;let mesh;
      if(n.type==='Mesh'){
        if(p.holeIndices?.length)throw Error('USD meshes with holes must be triangulated / 请先三角化带孔 USD 网格');const points=p.points?.map(v=>nums(v));if(!points?.length)throw Error('USD mesh has no points: '+n.path);const counts=nums(p.faceVertexCounts),indices=nums(p.faceVertexIndices),faces=[];let offset=0;
        for(const count of counts){if(!Number.isInteger(count)||count<3||count>4)throw Error('Browser USDA meshes require triangles or quads / 请先将 USDA 网格转换为三角面或四边面');const f=indices.slice(offset,offset+count);if(f.length!==count||f.some(i=>!Number.isInteger(i)||i<0||i>=points.length))throw Error('Invalid USD mesh face indices');for(let k=1;k+1<count;k++)faces.push([f[0],f[k],f[k+1]]);offset+=count;}if(offset!==indices.length)throw Error('USD face counts do not match indices');mesh={points,faces};if(p.subdivisionScheme&&p.subdivisionScheme!=='none')m.note('Subdivision meshes display their polygon cage. / 细分网格显示控制多边形。');if(p.orientation==='leftHanded')mesh.faces=mesh.faces.map(f=>f.slice().reverse());
      }else{
        const type=n.type.toLowerCase(),size=type==='cube'?[1,1,1].map(()=>number(p.size,2)):type==='sphere'?[number(p.radius,1)]:[number(p.radius,1),number(p.height,2)];mesh=primitive(type==='cube'?'box':type,size);
        if(['cylinder','capsule','cone'].includes(type)){const axis=p.axis||'Z',q=new T.Quaternion().setFromUnitVectors(V([0,0,1]),V(axis==='X'?[1,0,0]:axis==='Y'?[0,1,0]:[0,0,1]));mesh.points=mesh.points.map(v=>arr(V(v).applyQuaternion(q)));}
      }
      const color=nums(p['primvars:displayColor']?.[0],[.72,.78,.84]);color.push(number(p['primvars:displayOpacity']?.[0],1));const s=shape('mesh',[],I(),color);s.mesh={points:mesh.points.map(v=>point(relative,v)),faces:relative.determinant()<0?mesh.faces.map(f=>f.slice().reverse()):mesh.faces};
      const collision=p['urdf:role']==='collision'||(n.meta.apiSchemas||[]).includes('PhysicsCollisionAPI');if(collision)l.collisions.push(s);if(p['urdf:role']==='visual'||(!collision&&p.visibility!=='invisible'))l.visuals.push(s);else if(collision&&p.visibility!=='invisible'&&!p['urdf:role'])l.visuals.push(s);
    }
    if(bodies.length){
      const w=m.links.get(worldName),rootJoints=m.joints.filter(j=>j.parent===worldName),originalRoot=modelRoot?.props['urdf:rootLink'];
      const identity=j=>j.kind==='fixed'&&j.pose.elements.every((v,i)=>Math.abs(v-I().elements[i])<1e-9),anchor=rootJoints.find(j=>j.child===originalRoot&&identity(j));
      if(!w.visuals.length&&!w.collisions.length&&(anchor||(rootJoints.length===1&&identity(rootJoints[0])))){
        const remove=anchor||rootJoints[0];for(const j of rootJoints)if(j!==remove)j.parent=remove.child;m.links.delete(worldName);m.joints.splice(m.joints.indexOf(remove),1);
      }
    }
    if(all.some(n=>n.type==='Material'||n.type==='Shader'))m.note('USD materials reduce to displayColor; complex shaders and textures are not transferred. / USD 材质使用 displayColor，不迁移复杂着色器和贴图。');validate(m);return m;
  }
  function obj(mesh){return mesh.points.map(p=>'v '+fmt(p)).concat(mesh.faces.map(f=>'f '+f.map(i=>i+1).join(' '))).join('\n')+'\n';}
  function meshAssets(){const files=new Map(),cache=new Map();return {files,path(s){const mesh=geometryMesh(s);if(!cache.has(mesh)){const path='meshes/mesh_'+cache.size+'.obj';cache.set(mesh,path);files.set(path,obj(mesh));}return cache.get(mesh);}};}
  // Format generated XML through its DOM so escaped attribute values remain intact.
  function formattedXML(text){
    const lines=['<?xml version="1.0" encoding="utf-8"?>'];
    function element(node,depth){
      const indent='  '.repeat(depth),attrs=[...node.attributes].map(a=>` ${a.name}="${escape(a.value)}"`).join(''),tag=node.tagName;
      if(!node.children.length){lines.push(`${indent}<${tag}${attrs}/>`);return;}
      lines.push(`${indent}<${tag}${attrs}>`);for(const child of node.children)element(child,depth+1);lines.push(`${indent}</${tag}>`);
    }
    element(xml(text),0);return lines.join('\n')+'\n';
  }
  // Bound mesh-array rows without splitting a point tuple or changing its values.
  function usdArray(property,values){
    if(property.length+values.reduce((n,v)=>n+v.length+2,0)<96)return [`${property} = [${values.join(', ')}]`];
    const lines=[`${property} = [`];let row='';
    for(const value of values){if(row&&row.length+value.length+2>96){lines.push(row+',');row='';}row+=(row?', ':'')+value;}
    if(row)lines.push(row);lines.push(']');return lines;
  }
  function formattedUSD(lines){
    let depth=0;return lines.map(line=>{
      if(['}',')',']'].includes(line))depth--;
      const result=line?'    '.repeat(depth)+line:'';
      if(line.endsWith('{')||line==='('||line.endsWith(' = ['))depth++;
      return result;
    }).join('\n')+'\n';
  }
  function writeURDF(m){
    validate(m);const a=meshAssets(),lines=['<?xml version="1.0" encoding="utf-8"?>',`<robot name="${escape(m.name)}">`];
    for(const l of m.links.values()){lines.push(`<link name="${escape(l.name)}">`);if(l.mass>0)lines.push(`<inertial><origin xyz="${fmt(l.com)}"/><mass value="${l.mass}"/><inertia ${['ixx','iyy','izz','ixy','ixz','iyz'].map((k,i)=>k+'="'+tensorValues(l.inertia)[i]+'"').join(' ')}/></inertial>`);
      for(const [tag,shapes] of [['visual',l.visuals],['collision',l.collisions]])for(const s of shapes){let geo;if(s.kind==='box')geo=`<box size="${fmt(s.size)}"/>`;else if(s.kind==='sphere')geo=`<sphere radius="${s.size[0]}"/>`;else if(s.kind==='cylinder')geo=`<cylinder radius="${s.size[0]}" length="${s.size[1]}"/>`;else geo=`<mesh filename="${a.path(s)}"/>`;lines.push(`<${tag}>${origin(s.pose)}<geometry>${geo}</geometry>${tag==='visual'?`<material name="display"><color rgba="${fmt(s.color)}"/></material>`:''}</${tag}>`);}
      lines.push('</link>');
    }
    for(const j of m.joints){lines.push(`<joint name="${escape(j.name)}" type="${j.kind}"><parent link="${escape(j.parent)}"/><child link="${escape(j.child)}"/>${origin(j.pose)}`);if(!['fixed','floating'].includes(j.kind))lines.push(`<axis xyz="${fmt(j.axis)}"/><limit effort="${j.effort}" velocity="${j.velocity}"${j.kind!=='continuous'?` lower="${j.lower}" upper="${j.upper}"`:''}/><dynamics damping="${j.damping}" friction="${j.friction}"/>`);if(j.mimic)lines.push(`<mimic joint="${escape(j.mimic.joint)}" multiplier="${j.mimic.multiplier??1}" offset="${j.mimic.offset??0}"/>`);lines.push('</joint>');}
    lines.push('</robot>');a.files.set('model.urdf',formattedXML(lines.join('\n')));return {entry:'model.urdf',files:a.files};
  }
  function writeMJCF(m){
    const root=validate(m),a=meshAssets(),assets=[],motors=[],equalities=[],kids=name=>m.joints.filter(j=>j.parent===name),fixedJointNames=Object.fromEntries(m.joints.filter(j=>j.kind==='fixed').map(j=>[j.child,j.name])),rootLink=m.links.get(root),emptyRoot=!rootLink.mass&&!rootLink.visuals.length&&!rootLink.collisions.length;
    const metadata={rootName:root,worldRoot:emptyRoot?root:null,fixedJointNames,joints:Object.fromEntries(m.joints.map(j=>[j.name,{effort:j.effort,velocity:j.velocity,mimic:j.mimic}]))};
    function body(name,j=null,top=false){const l=m.links.get(name),frame=j?.pose||I(),q=quaternion(frame),lines=[`<body name="${escape(name)}" pos="${fmt(xyz(frame))}" quat="${fmt([q.w,q.x,q.y,q.z])}">`];
      if(j&&j.kind!=='fixed'){
        if(!(l.mass>0))throw Error('MJCF requires positive mass and inertia on moving link / MJCF 运动连杆需要正质量与有效惯量: '+name);
        if(j.kind==='floating'){if(!top)throw Error('MJCF free joints must be at the world root / 自由关节必须位于 MJCF 根层级');lines.push(`<freejoint name="${escape(j.name)}"/>`);}
        else lines.push(`<joint name="${escape(j.name)}" type="${j.kind==='prismatic'?'slide':'hinge'}" axis="${fmt(j.axis)}" limited="${j.kind==='continuous'?'false':'true'}"${j.kind==='continuous'?'':` range="${fmt([j.lower,j.upper])}"`} damping="${j.damping}" frictionloss="${j.friction}"/>`);
        if(j.effort>0&&j.kind!=='floating')motors.push(`<motor name="motor_${motors.length}" joint="${escape(j.name)}" gear="1" forcelimited="true" forcerange="${fmt([-j.effort,j.effort])}"/>`);
      }
      if(l.mass>0){const vals=principal(l.inertia).values;if(vals.some(x=>!(x>0))||Math.max(...vals)>vals.reduce((s,x)=>s+x,0)-Math.max(...vals)+1e-8)throw Error('Nonphysical / missing inertia tensor / 惯量张量无效或缺失: '+name);lines.push(`<inertial mass="${l.mass}" pos="${fmt(l.com)}" fullinertia="${fmt(tensorValues(l.inertia))}"/>`);}
      for(const [role,shapes] of [['visual',l.visuals],['collision',l.collisions]])for(const s of shapes){const q=quaternion(s.pose),color=role==='visual'?s.color:[...s.color.slice(0,3),0];let geo;
        if(s.kind==='box')geo=`type="box" size="${fmt(s.size.map(v=>v/2))}"`;else if(s.kind==='sphere')geo=`type="sphere" size="${s.size[0]}"`;else if(s.kind==='cylinder')geo=`type="cylinder" size="${fmt([s.size[0],s.size[1]/2])}"`;
        else{const path=a.path(s),id='mesh_'+path.match(/mesh_(\d+)/)[1];if(!assets.some(x=>x.includes('name="'+id+'"')))assets.push(`<mesh name="${id}" file="${path}"/>`);geo=`type="mesh" mesh="${id}"`;}
        lines.push(`<geom ${geo} pos="${fmt(xyz(s.pose))}" quat="${fmt([q.w,q.x,q.y,q.z])}" rgba="${fmt(color)}" group="${role==='visual'?2:3}" contype="${role==='visual'?0:1}" conaffinity="${role==='visual'?0:1}"/>`);
      }
      for(const k of kids(name))lines.push(body(k.child,k));lines.push('</body>');return lines.join('\n');
    }
    const bodies=emptyRoot?kids(root).map(j=>body(j.child,j,true)).join('\n'):body(root,null,true);
    for(const j of m.joints)if(j.mimic)equalities.push(`<joint joint1="${escape(j.name)}" joint2="${escape(j.mimic.joint)}" polycoef="${fmt([j.mimic.offset||0,j.mimic.multiplier??1,0,0,0])}"/>`);
    const text=`<?xml version="1.0" encoding="utf-8"?>\n<mujoco model="${escape(m.name)}"><compiler angle="radian" fusestatic="false" inertiafromgeom="false"/><custom><text name="urdf_viewer_metadata" data="${escape(JSON.stringify(metadata))}"/></custom><asset>${assets.join('\n')}</asset><worldbody>${bodies}</worldbody><actuator>${motors.join('\n')}</actuator><equality>${equalities.join('\n')}</equality></mujoco>`;
    a.files.set('model.xml',formattedXML(text));m.note('MJCF uses unit-gear torque motors; controllers and simulation settings are not reconstructed. / MJCF 使用单位传动比力矩电机，不重建原控制器与仿真设置。');return {entry:'model.xml',files:a.files};
  }
  function writeUSD(m){
    const root=validate(m),w=worlds(m),paths=new Map(),safe=s=>s.replace(/[^a-zA-Z0-9_]/g,'_'),str=JSON.stringify,tuple=a=>'('+a.map(v=>Number(Math.abs(v)<1e-12?0:v.toPrecision(10))).join(', ')+')',rotation=matrix=>{const q=quaternion(matrix);return tuple([q.w,q.x,q.y,q.z]);};
    const lines=['#usda 1.0','(','defaultPrim = "Robot"','metersPerUnit = 1','kilogramsPerUnit = 1','upAxis = "Z"',')','','def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {',`custom string urdf:modelName = ${str(m.name)}`,`custom string urdf:rootLink = ${str(root)}`,'def Xform "Links" {'];
    let i=0;for(const l of m.links.values()){
      const name='L'+i+++'_'+safe(l.name);paths.set(l.name,'/Robot/Links/'+name);lines.push(`def Xform ${str(name)} (prepend apiSchemas = ["PhysicsRigidBodyAPI"${l.mass>0?', "PhysicsMassAPI"':''}]) {`,`custom string urdf:name = ${str(l.name)}`,`double3 xformOp:translate = ${tuple(xyz(w.get(l.name)))}`,`quatf xformOp:orient = ${rotation(w.get(l.name))}`,'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]');
      const floating=m.joints.find(j=>j.child===l.name&&j.kind==='floating');if(floating)lines.push(`custom string urdf:floatingJointName = ${str(floating.name)}`);
      if(l.mass>0){const p=principal(l.inertia);if(p.values.some(v=>v<0))throw Error('Nonphysical inertia: '+l.name);lines.push(`float physics:mass = ${l.mass}`,`point3f physics:centerOfMass = ${tuple(l.com)}`,`float3 physics:diagonalInertia = ${tuple(p.values)}`,`quatf physics:principalAxes = ${tuple([p.q.w,p.q.x,p.q.y,p.q.z])}`);}else lines.push('bool physics:kinematicEnabled = true');
      for(const [role,shapes] of [['visual',l.visuals],['collision',l.collisions]])for(const [k,s] of shapes.entries()){
        const mesh=geometryMesh(s);lines.push(`def Mesh "${role}_${k}"${role==='collision'?' (prepend apiSchemas = ["PhysicsCollisionAPI"])':''} {`);
        for(const [property,values] of [['point3f[] points',mesh.points.map(p=>tuple(point(s.pose,p)))],['int[] faceVertexCounts',mesh.faces.map(()=>'3')],['int[] faceVertexIndices',mesh.faces.flat().map(String)]])for(const row of usdArray(property,values))lines.push(row);
        lines.push('uniform token subdivisionScheme = "none"',`color3f[] primvars:displayColor = [${tuple(s.color.slice(0,3))}]`,`float[] primvars:displayOpacity = [${s.color[3]}]`,`custom string urdf:role = "${role}"`);if(role==='collision')lines.push('token visibility = "invisible"');lines.push('}');
      }lines.push('}');
    }
    lines.push('}','def Xform "Joints" {');
    function frames(a,b){return [`point3f physics:localPos0 = ${tuple(xyz(a))}`,`quatf physics:localRot0 = ${rotation(a)}`,`point3f physics:localPos1 = ${tuple(xyz(b))}`,`quatf physics:localRot1 = ${rotation(b)}`];}
    for(const [index,j] of m.joints.entries()){if(j.kind==='floating')continue;const align=pose([0,0,0],new T.Quaternion().setFromUnitVectors(V([1,0,0]),V(j.axis).normalize()));lines.push(`def ${j.kind==='fixed'?'PhysicsFixedJoint':j.kind==='prismatic'?'PhysicsPrismaticJoint':'PhysicsRevoluteJoint'} "J${index}_${safe(j.name)}" {`,`custom string urdf:name = ${str(j.name)}`,`rel physics:body0 = <${paths.get(j.parent)}>`,`rel physics:body1 = <${paths.get(j.child)}>`,...frames(product(j.pose,align),align));
      if(j.kind!=='fixed'){lines.push('uniform token physics:axis = "X"');if(j.kind!=='continuous'){const f=j.kind==='prismatic'?1:180/Math.PI;lines.push(`float physics:lowerLimit = ${j.lower*f}`,`float physics:upperLimit = ${j.upper*f}`);}for(const key of ['velocity','effort','damping','friction'])lines.push(`custom double urdf:${key} = ${j[key]}`);if(j.mimic)lines.push(`custom string urdf:mimic = ${str(JSON.stringify(j.mimic))}`);}lines.push('}');
    }
    lines.push('def PhysicsFixedJoint "WorldAnchor" {',`custom string urdf:name = "__world_anchor"`,`rel physics:body1 = <${paths.get(root)}>`,...frames(I(),I()),'}','}','}');m.note('USD export is text .usda with standard UsdPhysics schemas; controllers and simulator tuning are not transferred. / USD 导出为标准 UsdPhysics 文本 .usda，不迁移控制器和仿真器调参。');return {entry:'model.usda',files:new Map([['model.usda',formattedUSD(lines)]])};
  }
  // ZIP STORE is supported by browsers and standard unzip tools; no CDN dependency.
  function zip(files){
    const table=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;table[n]=c>>>0;}
    const parts=[],central=[];let offset=0,centralSize=0;const encoder=new TextEncoder();
    for(const [path,value] of files){const name=encoder.encode(path),data=typeof value==='string'?encoder.encode(value):value;let c=0xffffffff;for(const b of data)c=table[(c^b)&255]^(c>>>8);const crc=(c^0xffffffff)>>>0,header=new Uint8Array(30),v=new DataView(header.buffer);v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,name.length,true);parts.push(header,name,data);
      const h=new Uint8Array(46),d=new DataView(h.buffer);d.setUint32(0,0x02014b50,true);d.setUint16(4,20,true);d.setUint16(6,20,true);d.setUint16(8,0x800,true);d.setUint32(16,crc,true);d.setUint32(20,data.length,true);d.setUint32(24,data.length,true);d.setUint16(28,name.length,true);d.setUint32(42,offset,true);central.push(h,name);centralSize+=46+name.length;offset+=30+name.length+data.length;
    }
    const end=new Uint8Array(22),e=new DataView(end.buffer);e.setUint32(0,0x06054b50,true);e.setUint16(8,files.size,true);e.setUint16(10,files.size,true);e.setUint32(12,centralSize,true);e.setUint32(16,offset,true);return new Blob([...parts,...central,end],{type:'application/zip'});
  }
  window.URDFBrowserFormats={
    async process(action,entry,files,target='urdf',sourceInfo,options={}){
      await new Promise(resolve=>setTimeout(resolve,0));options.signal?.throwIfAborted();const r=new Resources(entry,files,{...options,step:target==='step'}),file=files.get(r.entry);if(!file)throw Error('Missing model entry: '+entry);if(files.size>1500||[...files.values()].reduce((s,f)=>s+f.size,0)>384*1024*1024)throw Error('Model exceeds 384 MiB / 1500 files');
      const ext=file.name.split('.').pop().toLowerCase(),isUSD=['usd','usda','usdc','usdz'].includes(ext),text=isUSD?null:await file.text(),format=isUSD?'usd':xml(text).tagName==='robot'?'urdf':'mjcf';
      const model=await (format==='urdf'?readURDF(r):format==='mjcf'?readMJCF(r):readUSD(r));for(const note of sourceInfo?.warnings||[])model.note(note);
      if(target==='step'){r.cache.clear();return URDFStepExport.convert(model,{root:validate(model),geometryMesh,sourceInfo,format},options);}
      const result=target==='urdf'?writeURDF(model):target==='mjcf'?writeMJCF(model):target==='usd'?writeUSD(model):null;if(!result)throw Error('Unsupported export format: '+target);
      const report={sourceFormat:sourceInfo?.format||format,workingFormat:format,targetFormat:target,entry:result.entry,model:model.name,links:model.links.size,joints:model.joints.length,units:{length:'m',mass:'kg',angle:'rad'},upAxis:'Z',engine:'browser',scope:'Rigid-body tree geometry, joints and explicit inertial data; not a dynamics-equivalence guarantee.',warnings:model.notes};
      if(action==='import'){
        const assets=[];for(const [path,data] of result.files)if(path!==result.entry){const bytes=new TextEncoder().encode(data);let binary='';for(let i=0;i<bytes.length;i+=16384)binary+=String.fromCharCode(...bytes.subarray(i,i+16384));assets.push({path,data:btoa(binary)});}
        return new Response(JSON.stringify({urdf:result.files.get(result.entry),assets,report}),{headers:{'Content-Type':'application/json'}});
      }
      result.files.set('conversion-report.json',JSON.stringify(report,null,2));result.files.set('README.txt','Open '+result.entry+' with its meshes/ folder beside it.\nSee conversion-report.json for support limits.\nGenerated locally in your browser.\n');return new Response(zip(result.files),{headers:{'Content-Type':'application/zip','X-Conversion-Report':encodeURIComponent(JSON.stringify(report))}});
    },
    // Expose the same readers to the integration checks; UI imports still use File objects.
    parseUSDA,validate,principal
  };
})();
