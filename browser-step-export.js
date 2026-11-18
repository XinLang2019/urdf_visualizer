/* STEP AP214 assembly export. Runs locally; no CAD service or external dependency. */
(() => {
  'use strict';
  const pause=()=>new Promise(resolve=>setTimeout(resolve,0));
  const text=value=>"'"+String(value).replace(/['\\]|[^\x20-\x7e]/g,c=>{
    if(c==="'")return "''";
    if(c==='\\')return '\\\\';
    let hex='';for(let i=0;i<c.length;i++)hex+=c.charCodeAt(i).toString(16).toUpperCase().padStart(4,'0');
    return '\\X2\\'+hex+'\\X0\\';
  })+"'";
  function real(value){
    if(!Number.isFinite(value))throw Error('STEP: invalid coordinate / 坐标无效');
    const [mantissa,exponent]=Number(value.toPrecision(14)).toString().toUpperCase().split('E');
    return mantissa+(mantissa.includes('.')?'':'.')+(exponent===undefined?'':'E'+exponent);
  }
  const tuple=values=>'('+values.map(real).join(',')+')';

  async function topology(mesh){
    const points=[],faces=[],vertices=new Map(),edges=new Map();let skipped=0;
    const indices=mesh.points.map(p=>{
      if(p.length!==3||!p.every(Number.isFinite))throw Error('STEP: invalid mesh vertex / 网格顶点无效');
      const key=p.map(v=>Math.round(v*1e9)).join(',');
      if(!vertices.has(key)){vertices.set(key,points.length);points.push(p);}
      return vertices.get(key);
    });
    for(let k=0;k<mesh.faces.length;k++){
      if(k&&k%5000===0)await pause();
      const f=mesh.faces[k].map(i=>indices[i]);
      if(f.length!==3||f.some(i=>i===undefined))throw Error('STEP: invalid triangle / 三角面索引无效');
      const [a,b,c]=f.map(i=>points[i]),u=b.map((v,i)=>v-a[i]),v=c.map((x,i)=>x-a[i]);
      const cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
      const longest=Math.max(Math.hypot(...u),Math.hypot(...v),Math.hypot(...c.map((x,i)=>x-b[i])));
      // Nearly collinear triangles can make CAD readers inflate edge tolerances by
      // millimetres. Use the same 1 nm resolution as vertex welding, in metres.
      if(new Set(f).size<3||Math.hypot(...cross)<=longest*1e-9){skipped++;continue;}
      const index=faces.length;faces.push(f);
      for(let i=0;i<3;i++){
        const a=f[i],b=f[(i+1)%3],key=a<b?a+','+b:b+','+a;
        if(!edges.has(key))edges.set(key,[]);
        edges.get(key).push({index,direction:a<b});
      }
    }
    if(!faces.length)throw Error('STEP: mesh has no nondegenerate faces / 网格没有有效三角面');
    const components=[],flip=new Map();
    for(let start=0;start<faces.length;start++){
      if(flip.has(start))continue;
      const component={faces:[],closed:true,manifold:true,min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]},queue=[start];flip.set(start,false);
      for(let q=0;q<queue.length;q++){
        if(q&&q%5000===0)await pause();
        const index=queue[q],f=faces[index];component.faces.push(index);
        for(const i of f)for(let d=0;d<3;d++){component.min[d]=Math.min(component.min[d],points[i][d]);component.max[d]=Math.max(component.max[d],points[i][d]);}
        for(let i=0;i<3;i++){
          const a=f[i],b=f[(i+1)%3],uses=edges.get(a<b?a+','+b:b+','+a);
          if(uses.length!==2)component.closed=false;
          if(uses.length>2)component.manifold=false;
          for(const use of uses){
            if(use.index===index)continue;
            const reverse=flip.get(index)!==((a<b)===use.direction);
            if(!flip.has(use.index)){flip.set(use.index,reverse);queue.push(use.index);}
            else if(flip.get(use.index)!==reverse){component.closed=false;component.manifold=false;}
          }
        }
      }
      components.push(component);
    }
    for(const [index,reverse] of flip)if(reverse)faces[index].reverse();
    // Nested/intersecting shells may describe cavities. Keep them as surfaces rather
    // than inventing overlapping solids or filling a void in a mesh-derived part.
    const closed=components.filter(c=>c.closed).sort((a,b)=>a.min[0]-b.min[0]);let comparisons=0;
    for(let i=0;i<closed.length;i++)for(let j=i+1;j<closed.length&&closed[j].min[0]<=closed[i].max[0];j++){
      if(++comparisons%10000===0)await pause();
      const a=closed[i],b=closed[j];
      if(a.min.every((v,k)=>v<=b.max[k]&&a.max[k]>=b.min[k])){a.closed=false;b.closed=false;}
    }
    for(const component of components)if(component.closed){
      const origin=points[faces[component.faces[0]][0]];let volume=0;
      for(const index of component.faces){
        const [a,b,c]=faces[index].map(i=>points[i].map((v,k)=>v-origin[k]));
        volume+=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
      }
      if(Math.abs(volume)<1e-24)component.closed=false;
      else if(volume<0)for(const index of component.faces)faces[index].reverse();
    }
    return {points,faces,components,skipped};
  }

  async function write(model,{root,geometryMesh,output,onProgress}){
    const products=[],links=new Map(),geometryCache=new Map();let entityCount=0,occurrence=0,shapeCount=0,faceCount=0,surfaceCount=0;
    const header=['ISO-10303-21;','HEADER;',"FILE_DESCRIPTION(('URDF-Viewer assembly'),'2;1');",`FILE_NAME('model.step',${text(new Date().toISOString())},(''),(''),'URDF-Viewer','URDF-Viewer','');`,"FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",'ENDSEC;','DATA;'];
    output.text(header.join('\n')+'\n');
    const reserve=()=>'#'+(++entityCount);
    // STEP permits forward references and arbitrary entity order. Only assembly
    // representations are deferred; millions of geometry records go straight out.
    const put=(id,value)=>output.text(id+' = '+value+';\n');
    const add=value=>{const id=reserve();put(id,value);return id;};
    const application=add("APPLICATION_CONTEXT('automotive design')");
    add(`APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,${application})`);
    const productContext=add(`PRODUCT_CONTEXT('',${application},'mechanical')`),definitionContext=add(`PRODUCT_DEFINITION_CONTEXT('part definition',${application},'design')`);
    const millimetres=add('(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))');
    const radians=add('(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))'),steradians=add('(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())');
    const uncertainty=add(`UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),${millimetres},'distance_accuracy_value','')`);
    const cartesian=p=>add(`CARTESIAN_POINT('',${tuple(p.map(v=>v*1000))})`),direction=p=>add(`DIRECTION('',${tuple(p)})`);
    const origin=cartesian([0,0,0]),z=direction([0,0,1]),x=direction([1,0,0]),identity=add(`AXIS2_PLACEMENT_3D('',${origin},${z},${x})`);
    const placement=matrix=>{if(!matrix)return identity;const e=matrix.elements;return add(`AXIS2_PLACEMENT_3D('',${cartesian(e.slice(12,15))},${direction(e.slice(8,11))},${direction(e.slice(0,3))})`);};
    function product(name,description='',items=[],kind='SHAPE_REPRESENTATION'){
      const p=add(`PRODUCT(${text(name)},${text(name)},${text(description)},(${productContext}))`),formation=add(`PRODUCT_DEFINITION_FORMATION('','',${p})`);
      const definition=add(`PRODUCT_DEFINITION('design',${text(description)},${formation},${definitionContext})`),shape=add(`PRODUCT_DEFINITION_SHAPE('','',${definition})`);
      const context=add(`(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((${uncertainty})) GLOBAL_UNIT_ASSIGNED_CONTEXT((${millimetres},${radians},${steradians})) REPRESENTATION_CONTEXT('',''))`);
      const representation=reserve();add(`SHAPE_DEFINITION_REPRESENTATION(${shape},${representation})`);
      const result={name,definition,representation,context,items:[identity,...items],kind};products.push(result);return result;
    }
    function assemble(parent,child,name,description,matrix,label=child.name){
      const target=placement(matrix);if(target!==identity)parent.items.push(target);
      const usage=add(`NEXT_ASSEMBLY_USAGE_OCCURRENCE(${text(String(++occurrence))},${text(name)},${text(label)},${parent.definition},${child.definition},$)`);
      const shape=add(`PRODUCT_DEFINITION_SHAPE('',${text(description)},${usage})`),transform=add(`ITEM_DEFINED_TRANSFORMATION('','',${identity},${target})`);
      const relation=add(`(REPRESENTATION_RELATIONSHIP('','',${child.representation},${parent.representation}) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(${transform}) SHAPE_REPRESENTATION_RELATIONSHIP())`);
      add(`CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(${relation},${shape})`);
    }
    async function geometry(shape,name){
      const key=shape.mesh||JSON.stringify([shape.kind,shape.size]);if(geometryCache.has(key))return geometryCache.get(key);
      onProgress({stage:'topology',name});
      const mesh=await topology(geometryMesh(shape)),vertices=[],vertexPoints=[],faces=[],edgeCache=new Map();
      for(let i=0;i<mesh.points.length;i++){
        vertices.push(cartesian(mesh.points[i]));vertexPoints.push(add(`VERTEX_POINT('',${vertices[i]})`));
        if(i%2048===0){await output.flush();await pause();}
      }
      function edge(a,b){
        const forward=a<b,key=forward?a+','+b:b+','+a;
        if(!edgeCache.has(key)){
          const first=Math.min(a,b),last=Math.max(a,b),delta=mesh.points[last].map((v,k)=>v-mesh.points[first][k]),length=Math.hypot(...delta);
          const vector=add(`VECTOR('',${direction(delta.map(v=>v/length))},1.)`),line=add(`LINE('',${vertices[first]},${vector})`);
          edgeCache.set(key,add(`EDGE_CURVE('',${vertexPoints[first]},${vertexPoints[last]},${line},.T.)`));
        }
        return add(`ORIENTED_EDGE('',*,*,${edgeCache.get(key)},${forward?'.T.':'.F.'})`);
      }
      for(let i=0;i<mesh.faces.length;i++){
        if(i%1000===0){await output.flush();onProgress({stage:'writing',name,triangles:faceCount+i});await pause();}
        const indices=mesh.faces[i],loop=add(`EDGE_LOOP('',(${indices.map((a,k)=>edge(a,indices[(k+1)%3])).join(',')}))`),bound=add(`FACE_OUTER_BOUND('',${loop},.T.)`);
        const [a,b,c]=indices.map(v=>mesh.points[v]),u=b.map((v,k)=>v-a[k]),v=c.map((n,k)=>n-a[k]);
        const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],norm=Math.hypot(...normal),length=Math.hypot(...u);
        const axes=add(`AXIS2_PLACEMENT_3D('',${vertices[indices[0]]},${direction(normal.map(v=>v/norm))},${direction(u.map(v=>v/length))})`),plane=add(`PLANE('',${axes})`);
        faces.push(add(`ADVANCED_FACE('',(${bound}),${plane},.T.)`));
      }
      const items=[];
      for(const component of mesh.components){
        const refs=component.faces.map(i=>faces[i]);
        if(component.closed){const shell=add(`CLOSED_SHELL('',(${refs.join(',')}))`);items.push(add(`MANIFOLD_SOLID_BREP('',${shell})`));}
        else{
          const shells=component.manifold?[add(`OPEN_SHELL('',(${refs.join(',')}))`)]:refs.map(face=>add(`OPEN_SHELL('',(${face}))`));
          items.push(add(`SHELL_BASED_SURFACE_MODEL('',(${shells.join(',')}))`));surfaceCount++;
        }
      }
      if(mesh.skipped)model.note(`STEP: skipped ${mesh.skipped} degenerate triangles / 已跳过退化三角面: ${name}`);
      if(mesh.components.some(c=>!c.closed))model.note('STEP: open, nonmanifold or overlapping shells are exported as surfaces, without filling holes. / 开放、非流形或重叠壳导出为曲面，不自动补洞。');
      const part=product(name,'Mesh-derived geometry',items,mesh.components.every(c=>c.closed)?'ADVANCED_BREP_SHAPE_REPRESENTATION':'SHAPE_REPRESENTATION');
      geometryCache.set(key,part);shapeCount++;faceCount+=faces.length;await output.flush();return part;
    }
    const robot=product(model.name,'URDF assembly; zero joint positions; millimetres');
    for(const link of model.links.values())links.set(link.name,product(link.name,'URDF link'));
    assemble(robot,links.get(root),root,'URDF root link');
    for(const joint of model.joints)assemble(links.get(joint.parent),links.get(joint.child),joint.name,`URDF ${joint.kind} joint; axis ${joint.axis.join(' ')}; zero pose`,joint.pose);
    for(const link of model.links.values()){
      const shapes=link.visuals.length?link.visuals:link.collisions;
      if(!link.visuals.length&&link.collisions.length)model.note('STEP: links without visuals use collision geometry. / 无视觉几何的连杆使用碰撞几何。');
      for(let i=0;i<shapes.length;i++){
        const name=link.name+'/'+(link.visuals.length?'visual_':'collision_')+(i+1),part=await geometry(shapes[i],name);
        assemble(links.get(link.name),part,name,'URDF geometry origin',shapes[i].pose,name);await pause();
      }
    }
    if(!shapeCount)throw Error('STEP: model has no geometry / 模型没有可导出的几何');
    for(const p of products)put(p.representation,`${p.kind}(${text(p.name)},(${p.items.join(',')}),${p.context})`);
    model.note('STEP is a static AP214 assembly at URDF zero positions, in millimetres. Joint motion, limits and physics are not CAD constraints. / STEP 为毫米单位的 URDF 零位静态装配，关节运动、限位和物理参数不转为 CAD 约束。');
    model.note('Mesh surfaces remain faceted; original analytic CAD surfaces are not reconstructed. / 网格保持三角面，不恢复原始参数化 CAD 曲面。');
    output.text('ENDSEC;\nEND-ISO-10303-21;\n');await output.flush();
    return {schema:'AP214',lengthUnit:'mm',pose:'urdf-zero',geometryDefinitions:shapeCount,triangles:faceCount,surfaceComponents:surfaceCount};
  }

  const encoder=new TextEncoder(),memoryLimit=64*1024*1024;
  const crcTable=Uint32Array.from({length:256},(_,n)=>{let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;return c>>>0;});
  async function storage(name,estimatedBytes){
    let directory,handle,file;
    try{
      directory=await navigator.storage.getDirectory();
      // Recover only this exporter's abandoned files, never imported model data.
      for await(const [key,value] of directory.entries())if(key.startsWith('urdf-step-')&&key.endsWith('.zip')&&value.kind==='file'){
        const old=await value.getFile();if(Date.now()-old.lastModified>3600000)await directory.removeEntry(key).catch(()=>{});
      }
      const quota=await navigator.storage.estimate();
      if(quota.quota-quota.usage<estimatedBytes*1.15)throw Error('STEP_STORAGE_FULL');
      file=await directory.getFileHandle(name,{create:true});handle=await file.createSyncAccessHandle();
    }catch(error){
      if(handle)handle.close();if(directory&&file)await directory.removeEntry(name).catch(()=>{});
      if(error.message==='STEP_STORAGE_FULL')throw Error('STEP: insufficient browser temporary storage / 浏览器临时空间不足，请清理站点存储或先简化网格');
      if(estimatedBytes>memoryLimit)throw Error('STEP: temporary storage is unavailable; allow site storage on HTTPS / localhost, or simplify meshes first. / 临时存储不可用，请使用 HTTPS / localhost 并允许站点存储，或先简化网格');
      directory=null;handle=null;
    }
    let size=0;const parts=[];
    return {
      mode:handle?'disk':'memory',
      write(bytes){
        if(size+bytes.length>0x7fffffff||(!handle&&size+bytes.length>memoryLimit))throw Error('STEP: export exceeds available capacity; simplify meshes first / 导出超过可用容量，请先简化网格');
        if(handle){let offset=0;while(offset<bytes.length){const n=handle.write(bytes.subarray(offset),{at:size+offset});if(!n)throw Error('STEP: temporary file write failed / 临时文件写入失败');offset+=n;}}
        else parts.push(bytes);
        size+=bytes.length;
      },
      async finish(){if(handle){handle.flush();handle.close();handle=null;return file.getFile();}return new Blob(parts,{type:'application/zip'});},
      async abort(){if(handle){handle.close();handle=null;}if(directory)await directory.removeEntry(name).catch(()=>{});parts.length=0;}
    };
  }
  // ZIP STORE with data descriptors: CRC and sizes are computed as bytes are
  // written, without rereading STEP or materializing its text or byte buffer.
  function archive(sink){
    const central=[];let offset=0,current=null,pending=[],pendingSize=0;
    const append=bytes=>{sink.write(bytes);offset+=bytes.length;};
    const data=bytes=>{append(bytes);for(const b of bytes)current.crc=crcTable[(current.crc^b)&255]^(current.crc>>>8);current.size+=bytes.length;};
    return {
      begin(name){
        const path=encoder.encode(name),bytes=new Uint8Array(30+path.length),v=new DataView(bytes.buffer);
        v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x808,true);v.setUint16(26,path.length,true);bytes.set(path,30);
        current={path,offset,crc:0xffffffff,size:0};append(bytes);
      },
      text(value){pending.push(value);pendingSize+=value.length;},
      async flush(){if(pendingSize){data(encoder.encode(pending.join('')));pending=[];pendingSize=0;}},
      async end(){
        await this.flush();current.crc=(current.crc^0xffffffff)>>>0;
        const bytes=new Uint8Array(16),v=new DataView(bytes.buffer);v.setUint32(0,0x08074b50,true);v.setUint32(4,current.crc,true);v.setUint32(8,current.size,true);v.setUint32(12,current.size,true);append(bytes);
        central.push(current);current=null;
      },
      async finish(){
        const start=offset;
        for(const e of central){const bytes=new Uint8Array(46+e.path.length),v=new DataView(bytes.buffer);v.setUint32(0,0x02014b50,true);v.setUint16(4,20,true);v.setUint16(6,20,true);v.setUint16(8,0x808,true);v.setUint32(16,e.crc,true);v.setUint32(20,e.size,true);v.setUint32(24,e.size,true);v.setUint16(28,e.path.length,true);v.setUint32(42,e.offset,true);bytes.set(e.path,46);append(bytes);}
        const bytes=new Uint8Array(22),v=new DataView(bytes.buffer);v.setUint32(0,0x06054b50,true);v.setUint16(8,central.length,true);v.setUint16(10,central.length,true);v.setUint32(12,offset-start,true);v.setUint32(16,start,true);append(bytes);return sink.finish();
      }
    };
  }
  if(typeof document==='undefined'){
    self.onmessage=async event=>{
      let sink;
      try{
        const {model,root,sourceInfo,format,name}=event.data;
        const unique=new Set();for(const l of model.links.values())for(const s of l.visuals.length?l.visuals:l.collisions)unique.add(s.mesh);
        const total=[...unique].reduce((n,m)=>n+m.faces.length/3,0);
        sink=await storage(name,total*1100+65536);const zip=archive(sink);
        const onProgress=info=>self.postMessage({type:'progress',...info,total,progress:Math.min(.95,.15+.8*(info.triangles||0)/Math.max(1,total))});
        model.note=s=>{if(!model.notes.includes(s))model.notes.push(s);};
        const geometryMesh=shape=>({points:Array.from({length:shape.mesh.points.length/3},(_,i)=>Array.from(shape.mesh.points.subarray(i*3,i*3+3))),faces:Array.from({length:shape.mesh.faces.length/3},(_,i)=>Array.from(shape.mesh.faces.subarray(i*3,i*3+3)))});
        zip.begin('model.step');const details=await write(model,{root,geometryMesh,output:zip,onProgress});await zip.end();
        const report={sourceFormat:sourceInfo?.format||format,workingFormat:format,targetFormat:'step',entry:'model.step',model:model.name,links:model.links.size,joints:model.joints.length,units:{length:'mm',mass:'kg',angle:'rad'},upAxis:'Z',engine:'browser',scope:'Static mesh-derived CAD assembly at URDF zero positions; no kinematic CAD constraints or inertial data.',warnings:model.notes,step:{...details,storage:sink.mode,worker:true}};
        self.postMessage({type:'progress',stage:'packaging',progress:.98});
        zip.begin('conversion-report.json');zip.text(JSON.stringify(report,null,2));await zip.end();
        zip.begin('README.txt');zip.text('Open model.step in a STEP-compatible CAD application. Geometry is embedded.\nURDF zero-position assembly, millimetres, Z up. Joint motion is not exported as CAD constraints.\nMesh surfaces are faceted; open meshes remain surfaces. See conversion-report.json.\nGenerated locally in your browser.\n');await zip.end();
        const blob=await zip.finish();self.postMessage({type:'done',blob,report});
      }catch(error){if(sink)await sink.abort().catch(()=>{});self.postMessage({type:'error',message:error.message||String(error)});}
    };
  }else{
    const workerURL=document.currentScript.src;
    const abort=signal=>{if(signal?.aborted)throw new DOMException('Conversion cancelled','AbortError');};
    const cleanup=async name=>{
      for(let attempt=0;attempt<10;attempt++)try{const dir=await navigator.storage.getDirectory();await dir.removeEntry(name);return;}
      catch(error){if(error.name==='NotFoundError')return;await new Promise(resolve=>setTimeout(resolve,100));}
    };
    globalThis.URDFStepExport={async convert(model,{root,geometryMesh,sourceInfo,format},{signal,onProgress=()=>{}}={}){
      abort(signal);onProgress({stage:'preparing',progress:.1});
      const known=new WeakMap(),primitives=new Map(),meshes=[],transfer=[];
      for(const link of model.links.values())for(const shape of link.visuals.length?link.visuals:link.collisions){
        abort(signal);const primitiveKey=JSON.stringify([shape.kind,shape.size]);let packed=shape.mesh?known.get(shape.mesh):primitives.get(primitiveKey);
        if(!packed){
          const mesh=geometryMesh(shape),typed=ArrayBuffer.isView(mesh.points),pointCount=typed?mesh.points.length/3:mesh.points.length,faceCount=typed?mesh.faces.length/3:mesh.faces.length;
          if(faceCount>250000)throw Error('STEP: a mesh exceeds 250,000 triangles; simplify it with Mesh Simplifier first. / 单个网格超过 25 万面，请先用 Mesh Simplifier 简化');
          const points=typed?mesh.points:new Float64Array(pointCount*3),faces=typed?mesh.faces:new Uint32Array(faceCount*3);
          if(!typed){for(let i=0;i<pointCount;i++){points.set(mesh.points[i],i*3);if(i%4096===0){await pause();abort(signal);}}for(let i=0;i<faceCount;i++)faces.set(mesh.faces[i],i*3);}
          packed={points,faces};meshes.push(packed);transfer.push(points.buffer,faces.buffer);
          if(shape.mesh)known.set(shape.mesh,packed);else primitives.set(primitiveKey,packed);
          known.set(packed,packed);
        }
        shape.mesh=packed;await pause();
      }
      // Nonexported collisions can retain large arrays, so drop them before transfer.
      for(const link of model.links.values())if(link.visuals.length)link.collisions=[];
      abort(signal);const name='urdf-step-'+crypto.randomUUID()+'.zip';
      const data={model:{name:model.name,notes:model.notes,links:model.links,joints:model.joints},root,sourceInfo,format,name};
      return new Promise((resolve,reject)=>{
        let worker,done=false;
        const finish=(error,result)=>{if(done)return;done=true;signal?.removeEventListener('abort',cancel);worker?.terminate();if(error){cleanup(name);reject(error);}else resolve(result);};
        const cancel=()=>finish(new DOMException('Conversion cancelled','AbortError'));
        try{
          worker=new Worker(workerURL);signal?.addEventListener('abort',cancel,{once:true});abort(signal);
          worker.onmessage=event=>{
            const message=event.data;
            if(message.type==='progress')onProgress(message);
            else if(message.type==='error')finish(Error(message.message));
            else if(message.type==='done'){
              const response=new Response(message.blob,{headers:{'Content-Type':'application/zip','X-Conversion-Report':encodeURIComponent(JSON.stringify(message.report))}});
              response.downloadBlob=message.blob;response.cleanup=()=>cleanup(name);setTimeout(response.cleanup,300000);finish(null,response);
            }
          };
          worker.onerror=event=>{event.preventDefault();finish(Error('STEP worker stopped; simplify the mesh and retry / STEP 后台任务已停止，请简化网格后重试'));};
          worker.onmessageerror=()=>finish(Error('STEP: worker data transfer failed / 后台数据传输失败'));
          worker.postMessage(data,transfer);
        }catch(error){finish(error);}
      });
    }};
  }
})();
