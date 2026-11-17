/* Mesh parsing, welding, attribute-aware simplification and export stay off the UI thread. */
import { MeshoptSimplifier as simplify } from './vendor/meshoptimizer/meshopt_simplifier.js';

let source=null;
const progress=(stage,value)=>postMessage({type:'progress',stage,value});
const finite=a=>a.every(Number.isFinite);
function builder(info={}){return {...info,positions:[],normals:[],uvs:[],colors:[],indices:[],vertices:new Map()};}
function vertex(part,p,n=null,uv=null,color=null){
  if(!finite(p)||n&&!finite(n)||uv&&!finite(uv)||color&&!finite(color))throw Error('invalidNumbers');
  const key=[...p,...(n||[]),...(uv||[]),...(color||[])].map(x=>Math.fround(x)||0).join(',');
  let index=part.vertices.get(key);if(index===undefined){index=part.positions.length/3;part.vertices.set(key,index);part.positions.push(...p);if(n)part.normals.push(...n);if(uv)part.uvs.push(...uv);if(color)part.colors.push(...color);}return index;
}
function finish(part){delete part.vertices;for(const key of ['positions','normals','uvs','colors'])part[key]=new Float32Array(part[key]);part.indices=new Uint32Array(part.indices);return part;}
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
function compact(part,indices){
  const [remap,count]=simplify.compactMesh(indices),out={...part,indices};
  for(const key of ['positions','normals','uvs','colors']){const stride=key==='uvs'?2:3,old=part[key],a=new Float32Array(old.length?count*stride:0);if(old.length)for(let i=0;i<remap.length;i++)if(remap[i]!==0xffffffff)for(let k=0;k<stride;k++)a[remap[i]*stride+k]=old[i*stride+k];out[key]=a;}return out;
}
function simplifyPart(part,ratio,error,lockBorder){
  const count=part.positions.length/3,channels=[['normals',3,.5],['uvs',2,1],['colors',3,1]].filter(([key])=>part[key].length),stride=channels.reduce((s,[,n])=>s+n,0),target=Math.max(3,Math.floor(part.indices.length/3*ratio)*3),flags=lockBorder?['LockBorder']:[];
  let result;
  if(stride){const attributes=new Float32Array(count*stride),weights=[];let offset=0;for(const [key,n,weight] of channels){for(let i=0;i<count;i++)for(let k=0;k<n;k++)attributes[i*stride+offset+k]=part[key][i*n+k];weights.push(...Array(n).fill(weight));offset+=n;}
    result=simplify.simplifyWithAttributes(part.indices,part.positions,3,attributes,stride,weights,null,target,error,flags);
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
      if(data.file.size>384*1024*1024)throw Error('tooLarge');progress('reading',0);const buffer=await data.file.arrayBuffer();progress('parsing',.15);source=data.file.name.toLowerCase().endsWith('.stl')?parseSTL(buffer):parseOBJ(new TextDecoder().decode(buffer));source.format=data.file.name.split('.').pop().toLowerCase();const info=stats(source.parts);if(!info.faces)throw Error('emptyMesh');if(info.faces>8e6)throw Error('tooLarge');postMessage({type:'loaded',parts:source.parts,stats:info,bytes:data.file.size,libraries:source.libraries});
    }else if(data.action==='simplify'){
      if(!source)throw Error('emptyMesh');if(!(data.ratio>0&&data.ratio<1&&data.error>=0&&data.error<=1))throw Error('invalidSettings');
      const result=[];let error=0;for(let i=0;i<source.parts.length;i++){progress('simplifying',.2+.65*i/source.parts.length);const r=simplifyPart(source.parts[i],data.ratio,data.error,data.lockBorder);result.push(r.part);error=Math.max(error,r.error);}
      progress('exporting',.9);const blob=source.format==='stl'?exportSTL(result,source.header):exportOBJ(result,source.libraries);postMessage({type:'simplified',parts:result,stats:stats(result),error,blob});
    }
  }catch(e){postMessage({type:'error',message:e.message||String(e)});}
};
