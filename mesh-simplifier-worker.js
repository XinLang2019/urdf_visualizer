/* Mesh parsing, welding, attribute-aware simplification and export stay off the UI thread. */
import { MeshoptSimplifier as simplify } from './vendor/meshoptimizer/meshopt_simplifier.js';
import { parseXml } from './vendor/parse-xml/parse-xml.js';

let source=null;
const progress=(stage,value)=>postMessage({type:'progress',stage,value});
const finite=a=>a.every(Number.isFinite);
function builder(info={}){return {...info,positions:[],normals:[],uvs:[],colors:[],indices:[],vertices:new Map()};}
function vertex(part,p,n=null,uv=null,color=null,extra=[]){
  if(!finite(p)||n&&!finite(n)||uv&&!finite(uv)||color&&!finite(color)||!finite(extra))throw Error('invalidNumbers');
  const key=[...p,...(n||[]),...(uv||[]),...(color||[]),...extra].map(x=>Math.fround(x)||0).join(',');
  let index=part.vertices.get(key);if(index===undefined){index=part.positions.length/3;part.vertices.set(key,index);part.positions.push(...p);if(n)part.normals.push(...n);if(uv)part.uvs.push(...uv);if(color)part.colors.push(...color);}return index;
}
function finish(part){delete part.vertices;for(const key of ['positions','normals','uvs','colors'])part[key]=new Float32Array(part[key]);for(const a of part.daeAttributes||[])a.values=new Float32Array(a.values);part.indices=new Uint32Array(part.indices);return part;}
function triangle(part,a,b,c){if(a!==b&&b!==c&&c!==a)part.indices.push(a,b,c);}
function parseSTL(buffer){
  const view=new DataView(buffer),parts=new Map(),binary=buffer.byteLength>=84&&84+view.getUint32(80,true)*50===buffer.byteLength;
  const part=attribute=>{if(!parts.has(attribute))parts.set(attribute,builder({attribute,name:'mesh',material:'',smoothing:'off'}));return parts.get(attribute);};
  if(binary){const count=view.getUint32(80,true);if(count>8e6)throw Error('tooLarge');for(let i=0;i<count;i++){const offset=84+i*50,p=part(view.getUint16(offset+48,true)),indices=[];for(let j=0;j<3;j++)indices.push(vertex(p,[0,1,2].map(k=>view.getFloat32(offset+12+j*12+k*4,true))));triangle(p,...indices);}}
  else{const text=new TextDecoder().decode(buffer);if(!/^\s*solid\b/i.test(text))throw Error('invalidSTL');const pattern=/\bvertex\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)/g,p=part(0);let match,indices=[];while((match=pattern.exec(text))){indices.push(vertex(p,match.slice(1).map(Number)));if(indices.length===3){triangle(p,...indices);indices=[];}}if(indices.length)throw Error('invalidSTL');}
  return {parts:[...parts.values()].filter(p=>p.indices.length).map(finish),libraries:[],header:binary?new Uint8Array(buffer.slice(0,80)):null};
}
function triangulate(corners,positions){
  if(corners.length===3)return [[0,1,2]];if(corners.length>512)throw Error('complexPolygon');
  const pts=corners.map(c=>positions[c.v]),normal=[0,0,0];for(let i=0;i<pts.length;i++){const a=pts[i],b=pts[(i+1)%pts.length];normal[0]+=(a[1]-b[1])*(a[2]+b[2]);normal[1]+=(a[2]-b[2])*(a[0]+b[0]);normal[2]+=(a[0]-b[0])*(a[1]+b[1]);}
  const axis=normal.map(Math.abs).indexOf(Math.max(...normal.map(Math.abs))),axes=[0,1,2].filter(k=>k!==axis),p=pts.map(v=>axes.map(k=>v[k])),cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  let area=0;for(let i=0;i<p.length;i++)area+=p[i][0]*p[(i+1)%p.length][1]-p[(i+1)%p.length][0]*p[i][1];if(!area)throw Error('invalidPolygon');
  const sign=Math.sign(area),remaining=p.map((_,i)=>i),faces=[],eps=Math.abs(area)*1e-12;let passes=0;
  while(remaining.length>3){let found=false;for(let i=0;i<remaining.length;i++){const a=remaining[(i+remaining.length-1)%remaining.length],b=remaining[i],c=remaining[(i+1)%remaining.length];if(sign*cross(p[a],p[b],p[c])<=eps)continue;
      if(remaining.some(j=>j!==a&&j!==b&&j!==c&&sign*cross(p[a],p[b],p[j])>=-eps&&sign*cross(p[b],p[c],p[j])>=-eps&&sign*cross(p[c],p[a],p[j])>=-eps))continue;
      faces.push([a,b,c]);remaining.splice(i,1);found=true;break;}
    if(!found||++passes>corners.length*corners.length)throw Error('invalidPolygon');
  }faces.push(remaining);return faces;
}
function parseOBJ(text){
  const positions=[],normals=[],uvs=[],colors=[],parts=[],libraries=[];let name='mesh',group='',material='',smoothing='off',current=null;
  const index=(token,length)=>{const value=Number(token),i=value<0?length+value:value-1;if(!Number.isInteger(value)||value===0||i<0||i>=length)throw Error('invalidOBJIndex');return i;};
  for(const raw of text.replace(/\\\r?\n/g,' ').split(/\r?\n/)){
    const line=raw.trim();if(!line||line[0]==='#')continue;const split=line.search(/\s/),tag=split<0?line:line.slice(0,split),rest=split<0?'':line.slice(split).trim().split('#')[0].trim(),tokens=rest.split(/\s+/);
    if(tag==='v'){const v=tokens.map(Number);if(v.length<3||!finite(v))throw Error('invalidNumbers');if(v.length===4&&v[3]!==1)throw Error('weightedOBJ');positions.push(v.slice(0,3));colors.push(v.length>=6?v.slice(3,6):null);}
    else if(tag==='vn'){const v=tokens.map(Number);if(v.length<3||!finite(v))throw Error('invalidNumbers');const length=Math.hypot(...v.slice(0,3));normals.push(length?v.slice(0,3).map(x=>x/length):[0,0,0]);}
    else if(tag==='vt'){const v=tokens.map(Number);if(!finite(v))throw Error('invalidNumbers');if(v.length>2&&v[2]!==0)throw Error('volumeUV');uvs.push([v[0]||0,v[1]||0]);}
    else if(tag==='o'){name=rest||'mesh';current=null;}
    else if(tag==='g'){group=rest;current=null;}
    else if(tag==='usemtl'){material=rest;current=null;}
    else if(tag==='s'){smoothing=rest||'off';current=null;}
    else if(tag==='mtllib'){if(!libraries.includes(rest))libraries.push(rest);}
    else if(tag==='f'){
      if(tokens.length<3)throw Error('invalidPolygon');const corners=tokens.map(t=>{const v=t.split('/');return {v:index(v[0],positions.length),uv:v[1]?index(v[1],uvs.length):null,n:v[2]?index(v[2],normals.length):null};});
      const hasUV=corners.every(c=>c.uv!==null),hasNormal=corners.every(c=>c.n!==null),hasColor=corners.every(c=>colors[c.v]);if(corners.some(c=>(c.uv!==null)!==hasUV||(c.n!==null)!==hasNormal))throw Error('mixedAttributes');
      const signature=[hasUV,hasNormal,hasColor].join();if(!current||current.signature!==signature){current=builder({name,group,material,smoothing,signature});parts.push(current);}
      const ids=corners.map(c=>vertex(current,positions[c.v],hasNormal?normals[c.n]:null,hasUV?uvs[c.uv]:null,hasColor?colors[c.v]:null));for(const f of triangulate(corners,positions))triangle(current,...f.map(i=>ids[i]));
    }else if(['l','p','curv','curv2','surf'].includes(tag))throw Error('notTriangles');
  }
  return {parts:parts.filter(p=>p.indices.length).map(finish),libraries,header:null};
}
// Rewrite geometry only: scene instances, units, transforms, materials and image
// paths remain in the original document. Parsing and serialization stay in this worker.
const localName=node=>node?.name?.split(':').pop();
const elements=(node,name)=>node?.children?.filter(n=>n.type==='element'&&(!name||localName(n)===name))||[];
const child=(node,name)=>elements(node,name)[0];
const numbers=node=>{const text=node?.text.trim();if(!text)return [];const a=text.split(/\s+/).map(Number);if(!finite(a))throw Error('invalidNumbers');return a;};
const integer=(value,fallback)=>{const n=value===undefined?fallback:Number(value);if(!Number.isSafeInteger(n)||n<0)throw Error('invalidDAE');return n;};
const escapeXML=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
function parseDAE(text){
  let document;try{document=parseXml(text,{includeOffsets:true,maxElementDepth:256});}catch{throw Error('invalidDAE');}
  const root=elements(document)[0],ids=new Map();if(localName(root)!=='COLLADA')throw Error('invalidDAE');
  function collect(node){
    if(['controller','animation'].includes(localName(node)))throw Error('animatedDAE');
    const id=node.attributes?.id;if(id){if(ids.has(id))throw Error('invalidDAE');ids.set(id,node);}
    for(const c of elements(node))collect(c);
  }collect(root);
  const reference=url=>{if(!url?.startsWith('#')||!ids.has(url.slice(1)))throw Error('invalidDAE');return ids.get(url.slice(1));};
  const sources=new Map();
  function source(url){
    if(sources.has(url))return sources.get(url);
    const node=reference(url),accessor=child(child(node,'technique_common'),'accessor');
    if(localName(node)!=='source'||!accessor)throw Error('unsupportedDAE');
    const array=reference(accessor.attributes.source);if(localName(array)!=='float_array')throw Error('unsupportedDAE');
    const data=numbers(array),stride=integer(accessor.attributes.stride,1),offset=integer(accessor.attributes.offset,0),count=integer(accessor.attributes.count),params=elements(accessor,'param');
    const extent=params.length||stride;
    if(!stride||params.length>stride||integer(array.attributes.count)!==data.length||offset+(count?(count-1)*stride+extent:0)>data.length)throw Error('invalidDAE');
    if(params.some(p=>p.attributes.type!=='float'))throw Error('unsupportedDAE');
    const named=params.map((p,index)=>({name:p.attributes.name,index})).filter(p=>p.name);
    const result={count,width:named.length||stride,read(index){
      if(!Number.isSafeInteger(index)||index<0||index>=count)throw Error('invalidDAE');
      const start=offset+index*stride;return named.length?named.map(p=>data[start+p.index]):data.slice(start,start+stride);
    }};sources.set(url,result);return result;
  }
  const parts=[],meshes=[];let faces=0;
  for(const geometry of elements(child(root,'library_geometries'),'geometry')){
    const mesh=child(geometry,'mesh');if(!mesh)throw Error('unsupportedDAE');
    if(!geometry.attributes.id)throw Error('invalidDAE');
    const record={start:mesh.start,end:mesh.end,name:mesh.name,attributes:mesh.attributes,parts:[],extras:elements(mesh,'extra').map(e=>text.slice(e.start,e.end))};
    for(const primitive of elements(mesh).filter(n=>!['source','vertices','extra'].includes(localName(n)))){
      const kind=localName(primitive);if(!['triangles','polylist','polygons','tristrips','trifans'].includes(kind))throw Error('unsupportedDAE');
      const inputs=elements(primitive,'input'),channels=new Map();let stride=0;
      for(const input of inputs){
        const offset=integer(input.attributes.offset,0);stride=Math.max(stride,offset+1);
        if(input.attributes.semantic==='VERTEX'){
          const vertices=reference(input.attributes.source);if(localName(vertices)!=='vertices')throw Error('invalidDAE');
          for(const attr of elements(vertices,'input')){const semantic=attr.attributes.semantic,set=integer(attr.attributes.set,0);channels.set(semantic+':'+set,{semantic,set,offset,source:source(attr.attributes.source)});}
        }
      }
      for(const input of inputs)if(input.attributes.semantic!=='VERTEX'){
        const semantic=input.attributes.semantic,set=integer(input.attributes.set,0);channels.set(semantic+':'+set,{semantic,set,offset:integer(input.attributes.offset,0),source:source(input.attributes.source)});
      }
      const position=channels.get('POSITION:0'),attributes=[...channels.values()].filter(a=>a!==position);
      if(!position||position.source.width!==3||!stride)throw Error('invalidDAE');
      for(const a of attributes){
        const width=a.source.width;
        if(!['NORMAL','TEXCOORD','COLOR','TEXTANGENT','TEXBINORMAL','TANGENT','BINORMAL'].includes(a.semantic)||!width||width>4||a.semantic==='NORMAL'&&width!==3||a.semantic==='COLOR'&&width<3)throw Error('unsupportedDAE');
      }
      const part=builder({geometry:geometry.attributes.id,material:primitive.attributes.material||'',daeAttributes:attributes.map(a=>({semantic:a.semantic,set:a.set,width:a.source.width,values:[]}))});
      const append=(data,mode='polygon')=>{
        if(data.length%stride)throw Error('invalidDAE');const count=data.length/stride;if(count<3)throw Error('invalidPolygon');
        if(mode==='polygon'&&count>512)throw Error('complexPolygon');
        if(mode==='triangles'&&count%3)throw Error('invalidDAE');
        faces+=mode==='triangles'?count/3:count-2;if(faces>8e6)throw Error('tooLarge');
        const positions=[],indices=[];
        const normalIndex=attributes.findIndex(a=>a.semantic==='NORMAL'),uvIndex=attributes.findIndex(a=>a.semantic==='TEXCOORD'),colorIndex=attributes.findIndex(a=>a.semantic==='COLOR');
        for(let i=0;i<count;i++){
          const p=position.source.read(data[i*stride+position.offset]),values=attributes.map(a=>a.source.read(data[i*stride+a.offset])),before=part.positions.length;
          const n=values[normalIndex],uv=values[uvIndex],color=values[colorIndex];
          const id=vertex(part,p,n,uv?[uv[0],uv[1]||0]:null,color?.slice(0,3),values.flat());
          if(before!==part.positions.length)for(let a=0;a<values.length;a++)part.daeAttributes[a].values.push(...values[a]);
          if(mode==='polygon'){positions.push(p);indices.push(id);}
          else if(mode==='triangles'){indices.push(id);if(indices.length===3){triangle(part,...indices);indices.length=0;}}
          else if(mode==='trifans'){if(i<2)indices.push(id);else{triangle(part,indices[0],indices[1],id);indices[1]=id;}}
          else{if(i<2)indices.push(id);else{if(i%2)triangle(part,indices[1],indices[0],id);else triangle(part,indices[0],indices[1],id);indices[0]=indices[1];indices[1]=id;}}
        }
        if(mode==='polygon')for(const f of triangulate(positions.map((p,v)=>({v})),positions))triangle(part,...f.map(i=>indices[i]));
      };
      const lists=elements(primitive,'p').map(numbers);
      if(lists.some(a=>a.some(n=>!Number.isSafeInteger(n)||n<0)))throw Error('invalidDAE');
      const declared=integer(primitive.attributes.count);
      if(kind==='triangles'){
        if(lists.length!==1||lists[0].length!==declared*3*stride)throw Error('invalidDAE');if(declared)append(lists[0],'triangles');
      }else if(kind==='polylist'){
        const counts=numbers(child(primitive,'vcount'));if(lists.length!==1||counts.length!==declared||counts.some(n=>!Number.isSafeInteger(n)||n<3)||counts.reduce((s,n)=>s+n,0)*stride!==lists[0].length)throw Error('invalidDAE');
        let offset=0;for(const count of counts){append(lists[0].slice(offset,offset+count*stride));offset+=count*stride;}
      }else{
        if(elements(primitive,'ph').length)throw Error('unsupportedDAE');if(lists.length!==declared)throw Error('invalidDAE');for(const list of lists)append(list,kind==='polygons'?'polygon':kind);
      }
      if(part.indices.length){record.parts.push(parts.length);parts.push(finish(part));}
    }meshes.push(record);
  }
  // Keep the preview in scene space; export still uses original mesh coordinates.
  const preview={instances:[],unit:1,up:'Y_UP'},asset=child(root,'asset'),unit=child(asset,'unit');
  if(unit){preview.unit=Number(unit.attributes.meter);if(!(preview.unit>0&&Number.isFinite(preview.unit)))throw Error('invalidDAE');}
  preview.up=child(asset,'up_axis')?.text.trim()||'Y_UP';if(!['X_UP','Y_UP','Z_UP'].includes(preview.up))throw Error('invalidDAE');
  function visit(node,parents=[],stack=[]){
    if(stack.includes(node)||stack.length>128)throw Error('invalidDAE');
    const local=[];
    for(const c of elements(node)){
      const type=localName(c);if(['matrix','translate','rotate','scale'].includes(type)){
        const values=numbers(c),size={matrix:16,translate:3,rotate:4,scale:3}[type];if(values.length!==size||type==='rotate'&&Math.hypot(...values.slice(0,3))===0)throw Error('invalidDAE');local.push({type,values});
      }else if(['lookat','skew'].includes(type))throw Error('unsupportedDAE');
    }
    const transforms=[...parents,...local];
    for(const c of elements(node)){
      if(localName(c)==='instance_geometry'){const geometry=reference(c.attributes.url);if(localName(geometry)!=='geometry')throw Error('invalidDAE');preview.instances.push({geometry:geometry.attributes.id,transforms});}
      else if(localName(c)==='node')visit(c,transforms,[...stack,node]);
      else if(localName(c)==='instance_node'){const instance=reference(c.attributes.url);if(localName(instance)!=='node')throw Error('invalidDAE');visit(instance,transforms,[...stack,node]);}
    }
  }
  const instance=child(child(root,'scene'),'instance_visual_scene');
  if(instance){const scene=reference(instance.attributes.url);if(localName(scene)!=='visual_scene')throw Error('invalidDAE');visit(scene);}
  else for(const geometry of elements(child(root,'library_geometries'),'geometry'))preview.instances.push({geometry:geometry.attributes.id,transforms:[]});
  return {parts,libraries:[],preview,dae:{text,meshes,ids:new Set(ids.keys())}};
}
function exportDAE(parts,dae){
  const replacements=[];let serial=0;
  const unique=()=>{let id;do{id='urdf_viewer_simplified_'+serial++;}while(dae.ids.has(id));return id;};
  const format=a=>{const lines=[];for(let i=0;i<a.length;i+=12)lines.push('          '+Array.from(a.slice(i,i+12),n=>Number(n.toPrecision(9))).join(' '));return lines.join('\n');};
  for(const mesh of dae.meshes){
    const prefix=mesh.name.includes(':')?mesh.name.split(':')[0]+':':'',tag=name=>prefix+name;
    const lines=[`<${mesh.name}${Object.entries(mesh.attributes).map(([k,v])=>` ${k}="${escapeXML(v)}"`).join('')}>`],primitives=[],vertexId=unique(),positionId=unique();
    function emitSource(a,id){
      const arrayId=unique(),params=(a.semantic==='TEXCOORD'?['S','T','P','Q']:a.semantic==='COLOR'?['R','G','B','A']:['X','Y','Z','W']).slice(0,a.width);
      lines.push(`        <${tag('source')} id="${id}">`,`          <${tag('float_array')} id="${arrayId}" count="${a.values.length}">`,format(a.values),`          </${tag('float_array')}>`,`          <${tag('technique_common')}>`,`            <${tag('accessor')} source="#${arrayId}" count="${a.values.length/a.width}" stride="${a.width}">`,...params.map(p=>`              <${tag('param')} name="${p}" type="float"/>`),`            </${tag('accessor')}>`,`          </${tag('technique_common')}>`,`        </${tag('source')}>`);
    }
    // Keep position and attribute indices separate so flat normals / UV seams
    // do not inflate the exported position array with duplicate coordinates.
    const positions=[],positionMap=new Map(),positionIndices=new Map();
    for(const i of mesh.parts){
      const old=parts[i].positions,remap=new Uint32Array(old.length/3);
      for(let v=0;v<remap.length;v++){
        const point=old.slice(v*3,v*3+3),key=point.join(',');let index=positionMap.get(key);
        if(index===undefined){index=positions.length/3;positionMap.set(key,index);positions.push(...point);}remap[v]=index;
      }positionIndices.set(i,remap);
    }
    emitSource({semantic:'POSITION',width:3,values:new Float32Array(positions)},positionId);
    for(const index of mesh.parts){
      const part=parts[index],sourceIds=part.daeAttributes.map(a=>{const id=unique();emitSource(a,id);return id;}),stride=sourceIds.length?2:1,indices=new Uint32Array(part.indices.length*stride);
      const remap=positionIndices.get(index);
      for(let i=0;i<part.indices.length;i++){indices[i*stride]=remap[part.indices[i]];if(stride===2)indices[i*stride+1]=part.indices[i];}
      primitives.push(`        <${tag('triangles')} count="${part.indices.length/3}"${part.material?` material="${escapeXML(part.material)}"`:''}>`,`          <${tag('input')} semantic="VERTEX" source="#${vertexId}" offset="0"/>`,...part.daeAttributes.map((a,i)=>`          <${tag('input')} semantic="${a.semantic}" source="#${sourceIds[i]}" offset="1" set="${a.set}"/>`),`          <${tag('p')}>`,format(indices),`          </${tag('p')}>`,`        </${tag('triangles')}>`);
    }
    lines.push(`        <${tag('vertices')} id="${vertexId}">`,`          <${tag('input')} semantic="POSITION" source="#${positionId}"/>`,`        </${tag('vertices')}>`);
    lines.push(...primitives,...mesh.extras,`      </${mesh.name}>`);replacements.push({...mesh,value:lines.join('\n')});
  }
  // Slice original text in ascending order: preserve everything outside meshes verbatim.
  const chunks=[];let offset=0;for(const r of replacements.sort((a,b)=>a.start-b.start)){chunks.push(dae.text.slice(offset,r.start),r.value);offset=r.end;}chunks.push(dae.text.slice(offset));
  return new Blob(chunks,{type:'model/vnd.collada+xml;charset=utf-8'});
}
function compact(part,indices){
  const [remap,count]=simplify.compactMesh(indices),out={...part,indices};
  const remapAttribute=(old,stride)=>{const a=new Float32Array(old.length?count*stride:0);if(old.length)for(let i=0;i<remap.length;i++)if(remap[i]!==0xffffffff)for(let k=0;k<stride;k++)a[remap[i]*stride+k]=old[i*stride+k];return a;};
  for(const key of ['positions','normals','uvs','colors'])out[key]=remapAttribute(part[key],key==='uvs'?2:3);
  if(part.daeAttributes)out.daeAttributes=part.daeAttributes.map(a=>({...a,values:remapAttribute(a.values,a.width)}));return out;
}
function simplifyPart(part,ratio,error,lockBorder){
  const count=part.positions.length/3,channels=part.daeAttributes?part.daeAttributes.map(a=>[a.values,a.width,a.semantic==='NORMAL'?.5:1]):[['normals',3,.5],['uvs',2,1],['colors',3,1]].filter(([key])=>part[key].length).map(([key,n,w])=>[part[key],n,w]),stride=channels.reduce((s,[,n])=>s+n,0),target=Math.max(3,Math.floor(part.indices.length/3*ratio)*3),flags=lockBorder?['LockBorder']:[];
  let result;
  if(stride>32)throw Error('unsupportedDAE');
  let locks=null;
  if(part.daeAttributes&&stride){
    // CAD exports often split every flat-shaded face at its normal seams.
    // Allow normal changes within the error metric while protecting UV/color seams.
    flags.push('Permissive');
    const protectedAttributes=part.daeAttributes.filter(a=>['TEXCOORD','COLOR'].includes(a.semantic));
    if(protectedAttributes.length){
      locks=new Uint8Array(count);const positions=new Map();
      for(let i=0;i<count;i++){
        const key=part.positions.slice(i*3,i*3+3).join(','),signature=protectedAttributes.map(a=>a.values.slice(i*a.width,(i+1)*a.width).join(',')).join(';'),other=positions.get(key);
        if(!other)positions.set(key,{signature,seam:false});
        else if(signature!==other.signature)other.seam=true;
      }
      for(let i=0;i<count;i++)if(positions.get(part.positions.slice(i*3,i*3+3).join(',')).seam)locks[i]=2;
    }
  }
  if(stride){const attributes=new Float32Array(count*stride),weights=[];let offset=0;for(const [values,n,weight] of channels){for(let i=0;i<count;i++)for(let k=0;k<n;k++)attributes[i*stride+offset+k]=values[i*n+k];weights.push(...Array(n).fill(weight));offset+=n;}
    result=simplify.simplifyWithAttributes(part.indices,part.positions,3,attributes,stride,weights,locks,target,error,flags);
  }else result=simplify.simplify(part.indices,part.positions,3,target,error,flags);
  if(!result[0].length)throw Error('emptyResult');return {part:compact(part,result[0]),error:result[1]};
}
function normal(a,b,c){const u=b.map((x,k)=>x-a[k]),v=c.map((x,k)=>x-a[k]),n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],length=Math.hypot(...n);return length?n.map(x=>x/length):[0,0,0];}
function exportSTL(parts,header){
  const faces=parts.reduce((s,p)=>s+p.indices.length/3,0),data=new Uint8Array(84+faces*50),view=new DataView(data.buffer);if(header)data.set(header);else data.set(new TextEncoder().encode('URDF-Viewer Mesh Simplifier'));view.setUint32(80,faces,true);let offset=84;
  for(const part of parts)for(let i=0;i<part.indices.length;i+=3){const points=[0,1,2].map(k=>[...part.positions.slice(part.indices[i+k]*3,part.indices[i+k]*3+3)]),n=normal(...points);for(const [j,v] of [...n,...points.flat()].entries())view.setFloat32(offset+j*4,v,true);view.setUint16(offset+48,part.attribute||0,true);offset+=50;}
  return new Blob([data],{type:'model/stl'});
}
function exportOBJ(parts,libraries){
  const chunks=['# Simplified by URDF-Viewer; original mesh coordinates retained.\n',...libraries.map(name=>'mtllib '+name+'\n')];let vOffset=1,nOffset=1,tOffset=1;
  const format=a=>a.map(x=>Number(x.toPrecision(9))).join(' ');
  for(const p of parts){const lines=['o '+p.name];if(p.group)lines.push('g '+p.group);if(p.material)lines.push('usemtl '+p.material);lines.push('s '+p.smoothing);
    for(let i=0;i<p.positions.length;i+=3)lines.push('v '+format([...p.positions.slice(i,i+3),...(p.colors.length?p.colors.slice(i,i+3):[])]));for(let i=0;i<p.uvs.length;i+=2)lines.push('vt '+format([...p.uvs.slice(i,i+2)]));for(let i=0;i<p.normals.length;i+=3)lines.push('vn '+format([...p.normals.slice(i,i+3)]));
    for(let i=0;i<p.indices.length;i+=3)lines.push('f '+[0,1,2].map(k=>{const id=p.indices[i+k];return String(vOffset+id)+(p.uvs.length||p.normals.length?'/'+(p.uvs.length?tOffset+id:'')+(p.normals.length?'/'+(nOffset+id):''):'');}).join(' '));
    chunks.push(lines.join('\n')+'\n');vOffset+=p.positions.length/3;tOffset+=p.uvs.length/2;nOffset+=p.normals.length/3;
  }return new Blob(chunks,{type:'text/plain;charset=utf-8'});
}
const stats=parts=>({faces:parts.reduce((s,p)=>s+p.indices.length/3,0),vertices:parts.reduce((s,p)=>s+p.positions.length/3,0)});
self.onmessage=async({data})=>{
  try{
    if(!simplify.supported)throw Error('wasmUnsupported');await simplify.ready;
    if(data.action==='load'){
      if(data.file.size>384*1024*1024)throw Error('tooLarge');progress('reading',0);const buffer=await data.file.arrayBuffer();progress('parsing',.15);const format=data.file.name.split('.').pop().toLowerCase();source=format==='stl'?parseSTL(buffer):format==='dae'?parseDAE(new TextDecoder().decode(buffer)):parseOBJ(new TextDecoder().decode(buffer));source.format=format;const info=stats(source.parts);if(!info.faces)throw Error('emptyMesh');if(info.faces>8e6)throw Error('tooLarge');postMessage({type:'loaded',parts:source.parts,stats:info,bytes:data.file.size,libraries:source.libraries,preview:source.preview});
    }else if(data.action==='simplify'){
      if(!source)throw Error('emptyMesh');if(!(data.ratio>0&&data.ratio<1&&data.error>=0&&data.error<=1))throw Error('invalidSettings');
      const result=[];let error=0;for(let i=0;i<source.parts.length;i++){progress('simplifying',.2+.65*i/source.parts.length);const r=simplifyPart(source.parts[i],data.ratio,data.error,data.lockBorder);result.push(r.part);error=Math.max(error,r.error);}
      progress('exporting',.9);const blob=source.format==='stl'?exportSTL(result,source.header):source.format==='dae'?exportDAE(result,source.dae):exportOBJ(result,source.libraries);postMessage({type:'simplified',parts:result,stats:stats(result),error,blob});
    }
  }catch(e){postMessage({type:'error',message:e.message||String(e)});}
};
