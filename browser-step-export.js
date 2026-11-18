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

  window.URDFStepExport={async write(model,{root,geometryMesh}){
    const records=[],products=[],links=new Map(),geometryCache=new Map();let occurrence=0,shapeCount=0,faceCount=0,surfaceCount=0;
    const reserve=()=>{records.push(null);return '#'+records.length;};
    const put=(id,value)=>{records[Number(id.slice(1))-1]=id+' = '+value+';';};
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
      const mesh=await topology(geometryMesh(shape)),vertices=mesh.points.map(cartesian),vertexPoints=vertices.map(v=>add(`VERTEX_POINT('',${v})`)),faces=[],edgeCache=new Map();
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
        if(i&&i%5000===0)await pause();
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
      geometryCache.set(key,part);shapeCount++;faceCount+=faces.length;return part;
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
    const header=['ISO-10303-21;','HEADER;',"FILE_DESCRIPTION(('URDF-Viewer assembly'),'2;1');",`FILE_NAME('model.step',${text(new Date().toISOString())},(''),(''),'URDF-Viewer','URDF-Viewer','');`,"FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",'ENDSEC;','DATA;'];
    return {entry:'model.step',files:new Map([['model.step',header.concat(records,['ENDSEC;','END-ISO-10303-21;','']).join('\n')]]),details:{schema:'AP214',lengthUnit:'mm',pose:'urdf-zero',geometryDefinitions:shapeCount,triangles:faceCount,surfaceComponents:surfaceCount}};
  }};
})();
