"""Real-worker COLLADA round trips, scene preservation and filesystem overwrite."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import browser_smoke as smoke
from mesh_layout_cache import import_meshes
from model_interaction import rendered_frame

ROOT=Path(__file__).resolve().parents[1]
NS={'c':'http://www.collada.org/2005/11/COLLADASchema'}
ROBOT='<robot name="dae_fixture"><link name="base"/><link name="arm"><visual><geometry><mesh filename="meshes/model.dae"/></geometry></visual><collision><geometry><mesh filename="meshes/model.dae"/></geometry></collision></link><joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="2" velocity="2"/></joint></robot>'


def probe(client,text,simplify=False):
    return client.js('''(async()=>{const path=document.querySelector('script[data-mesh-worker]').dataset.meshWorker,w=new Worker(path,{type:'module'});try{return await new Promise((resolve,reject)=>{let loaded;w.onerror=e=>reject(Error(e.message));w.onmessage=async({data:d})=>{if(d.type==='progress')return;if(d.type==='error')return resolve({error:d.message});if(d.type==='loaded'){loaded=d;if(!SIMPLIFY)return resolve({faces:d.stats.faces,parts:d.parts.length});w.postMessage({action:'simplify',ratio:.5,error:.05,lockBorder:false});}else if(d.type==='simplified')resolve({before:loaded.stats.faces,after:d.stats.faces,text:await d.blob.text()});};w.postMessage({action:'load',file:new File([TEXT],'probe.dae')});});}finally{w.terminate();}})()'''.replace('TEXT',json.dumps(text)).replace('SIMPLIFY',json.dumps(simplify)))


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1000,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    check=lambda expr,msg:smoke.check(client,expr,msg)
    client.js(r'''(()=>{
      const T=parcelRequire('../node_modules/three/build/three.module.js'),g=new T.SphereGeometry(1,32,24),p=g.attributes.position,n=g.attributes.normal,uv=g.attributes.uv,idx=g.index;
      const source=(id,data,stride,params)=>`<source id="${id}"><float_array id="${id}_data" count="${data.length}">${data.join(' ')}</float_array><technique_common><accessor source="#${id}_data" count="${data.length/stride}" stride="${stride}">${params.map(name=>`<param ${name?'name="'+name+'" ':''}type="float"/>`).join('')}</accessor></technique_common></source>`;
      const pos=[],norm=[],tex=[],tex2=[],colors=[];
      for(let i=0;i<p.count;i++){pos.push(p.getX(i)+2,p.getY(i)+3,p.getZ(i)+4);norm.push(n.getX(i),n.getY(i),n.getZ(i));tex.push(uv.getX(i),uv.getY(i));tex2.push(uv.getX(i)*.5,uv.getY(i)*.5);colors.push(1,.3,.2,.65)}
      const half=Math.floor(idx.count/6)*3,tri=(start,end,mat)=>`<triangles count="${(end-start)/3}" material="${mat}"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="COLOR" source="#c" offset="1"/><input semantic="NORMAL" source="#n" offset="2"/><input semantic="TEXCOORD" source="#t" offset="3" set="0"/><input semantic="TEXCOORD" source="#t2" offset="4" set="1"/><p>${Array.from({length:end-start},(_,i)=>Array(5).fill(idx.getX(start+i)).join(' ')).join(' ')}</p></triangles>`;
      const binding='<bind_material><technique_common><instance_material symbol="red" target="#red_mat"><bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/></instance_material><instance_material symbol="blue" target="#blue_mat"/></technique_common></bind_material>';
      window.daeTest={original:`<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<!-- 中文机器人 🤖: source offsets must survive astral Unicode. -->
<asset><contributor><author>Mesh test</author></contributor><created>2026-01-01T00:00:00Z</created><modified>2026-01-01T00:00:00Z</modified><unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis></asset>
<library_images><image id="image"><init_from>textures/pixel.png</init_from></image></library_images>
<library_effects><effect id="red_fx"><profile_COMMON><newparam sid="surface"><surface type="2D"><init_from>image</init_from></surface></newparam><newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam><technique sid="common"><phong><diffuse><texture texture="sampler" texcoord="UVMap"/></diffuse></phong></technique></profile_COMMON></effect><effect id="blue_fx"><profile_COMMON><technique sid="common"><phong><diffuse><color>0 0 1 1</color></diffuse></phong></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="red_mat"><instance_effect url="#red_fx"/></material><material id="blue_mat"><instance_effect url="#blue_fx"/></material></library_materials>
<library_geometries><geometry id="sphere" name="Sphere"><mesh>${source('p',pos,3,['X','Y','Z'])}${source('n',norm,3,['X','Y','Z'])}${source('t',tex,2,['S','T'])}${source('t2',tex2,2,['S','T'])}${source('c',colors,4,['R','G','B','A'])}<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>${tri(0,half,'red')}${tri(half,idx.count,'blue')}<extra><technique profile="fixture"><keep>mesh metadata</keep></technique></extra></mesh></geometry>
<geometry id="marker"><mesh>${source('marker_p',[0,0,0,1,0,0,0,1,0],3,['X','Y','Z'])}<vertices id="marker_v"><input semantic="POSITION" source="#marker_p"/></vertices><triangles count="1"><input semantic="VERTEX" source="#marker_v" offset="0"/><p>0 1 2</p></triangles></mesh></geometry></library_geometries>
<library_nodes><node id="instanced"><translate>0 3 0</translate><instance_geometry url="#sphere">${binding}</instance_geometry></node></library_nodes>
<library_visual_scenes><visual_scene id="scene"><node id="parent"><translate>1 2 3</translate><rotate>0 0 1 30</rotate><node id="first"><matrix>1 0 0 2 0 2 0 1 0 0 1 0 0 0 0 1</matrix><instance_geometry url="#sphere">${binding}</instance_geometry></node><instance_node url="#instanced"/><node><scale>2 1 1</scale><instance_geometry url="#marker"/></node></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/></scene></COLLADA>`,downloads:[]};g.dispose();
      const nativeClick=HTMLAnchorElement.prototype.click;
      HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.text()).then(text=>daeTest.downloads.push({name,text}))}else nativeClick.call(this)};
      daeTest.put=async(h,data)=>{const s=await h.createWritable();await s.write(data);await s.close()};
      daeTest.withoutMeshes=text=>{const d=new DOMParser().parseFromString(text,'application/xml');for(const m of d.querySelectorAll('mesh'))m.replaceChildren();return new XMLSerializer().serializeToString(d)};
      daeTest.bounds=scene=>{scene.updateMatrixWorld(true);const b=new T.Box3().setFromObject(scene);return b.getSize(new T.Vector3()).toArray()};
    })()''')
    original=client.js('daeTest.original')
    # Compare the actual preview builder against Three's independent COLLADA loader.
    client.js('''(async()=>{const w=new Worker(document.querySelector('script[data-mesh-worker]').dataset.meshWorker,{type:'module'});try{const data=await new Promise((resolve,reject)=>{w.onmessage=({data:d})=>{if(d.type==='error')reject(Error(d.message));else if(d.type==='loaded')resolve(d)};w.postMessage({action:'load',file:new File([daeTest.original],'model.dae')})});const T=parcelRequire('../node_modules/three/build/three.module.js'),C=parcelRequire('../node_modules/three/examples/jsm/loaders/ColladaLoader.js').ColladaLoader,preview=URDFMeshSimplifier.prototype.group.call({T,preview:data.preview},data.parts),native=new C().parse(daeTest.original,'').scene;daeTest.previewBounds=daeTest.bounds(preview);daeTest.nativeBounds=daeTest.bounds(native);daeTest.before=data.stats.faces;preview.traverse(o=>{o.geometry?.dispose();o.material?.dispose()});}finally{w.terminate()}})()''')
    check('daeTest.previewBounds.every((n,i)=>Math.abs(n-daeTest.nativeBounds[i])<1e-6)','DAE preview preserves units, up axis, nested transforms, matrices and shared geometry instances')
    # Use real writable folder handles, so overwrite and robot refresh are exercised.
    client.js('(async()=>{daeTest.dir=await(await navigator.storage.getDirectory()).getDirectoryHandle("dae-test",{create:true});const m=await daeTest.dir.getDirectoryHandle("meshes",{create:true});daeTest.handle=await m.getFileHandle("model.dae",{create:true});daeTest.robot=await daeTest.dir.getFileHandle("robot.urdf",{create:true});await daeTest.put(daeTest.handle,daeTest.original);await daeTest.put(daeTest.robot,'+json.dumps(ROBOT)+');const tex=await m.getDirectoryHandle("textures",{create:true});await daeTest.put(await tex.getFileHandle("pixel.png",{create:true}),await(await fetch("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==")).blob());window.showDirectoryPicker=async()=>daeTest.dir;document.getElementById("load-folder").click()})()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="dae_fixture"&&!urdfWorkbench.loading',45)
    check('urdfWorkbench.meshSimplifier.files.length===1&&urdfWorkbench.meshSimplifier.files[0].endsWith("model.dae")','DAE appears in the loaded mesh selector')
    client.js('urdfWorkbench.viewer.setJointValue("hinge",.4);document.querySelector(".file-entry[data-path$=\\"model.dae\\"]").click()')
    client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    check('document.getElementById("mesh-dialog").open&&urdfWorkbench.meshSimplifier.original.faces===daeTest.before','clicking DAE in the robot file tree opens the simplifier')
    client.js('document.getElementById("mesh-error").value=5;document.getElementById("mesh-error").dispatchEvent(new Event("change"));document.getElementById("mesh-run").click()')
    client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy',45)
    check('urdfWorkbench.meshSimplifier.result.faces<daeTest.before*.85','attribute-aware DAE simplification reduces triangles')
    check('(async()=>await(await daeTest.handle.getFile()).text()===daeTest.original)()','DAE preview and simplification leave the source file intact')
    client.js('document.getElementById("mesh-download").click()');client.until('daeTest.downloads.length===1')
    check('daeTest.downloads[0].name==="model.simplified.dae"&&daeTest.withoutMeshes(daeTest.downloads[0].text)===daeTest.withoutMeshes(daeTest.original)','DAE download preserves the complete scene, materials, textures, unit and metadata outside meshes')
    text=client.js('daeTest.downloads[0].text');(output/'model.simplified.dae').write_text(text)
    xml=ET.fromstring(text)
    assert len(xml.findall('.//c:mesh/c:vertices',NS))==2
    assert sum(int(e.get('count')) for e in xml.findall('.//c:triangles',NS))==client.js('urdfWorkbench.meshSimplifier.result.faces')
    assert len(xml.findall('.//c:mesh/c:extra',NS))==1
    inputs=xml.findall('.//c:triangles/c:input',NS)
    assert any(e.get('semantic')=='TEXCOORD' and e.get('set')=='1' for e in inputs)
    assert any(e.get('semantic')=='COLOR' for e in inputs)
    for color in [e for e in inputs if e.get('semantic')=='COLOR']:
        source=xml.find('.//c:source[@id="'+color.get('source')[1:]+'"]',NS)
        accessor=source.find('c:technique_common/c:accessor',NS)
        assert accessor.get('stride')=='4'
        values=list(map(float,source.find('c:float_array',NS).text.split()))
        assert all(abs(a-.65)<1e-6 for a in values[3::4])
    ids={e.get('id'):e for e in xml.iter() if e.get('id')}
    for triangles in xml.findall('.//c:triangles',NS):
        channels=triangles.findall('c:input',NS);stride=max(int(e.get('offset','0')) for e in channels)+1
        indices=list(map(int,triangles.find('c:p',NS).text.split()))
        assert len(indices)==int(triangles.get('count'))*3*stride
        for channel in channels:
            source=ids[channel.get('source')[1:]]
            if channel.get('semantic')=='VERTEX':
                source=ids[source.find('c:input',NS).get('source')[1:]]
            count=int(source.find('c:technique_common/c:accessor',NS).get('count'))
            assert all(0<=i<count for i in indices[int(channel.get('offset','0'))::stride])
    check('(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),C=parcelRequire("../node_modules/three/examples/jsm/loaders/ColladaLoader.js").ColladaLoader,s=new C().parse(daeTest.downloads[0].text,"").scene;return daeTest.bounds(s).every((n,i)=>Math.abs(n-daeTest.nativeBounds[i])<.005)})()','downloaded DAE reloads in the robot COLLADA loader with the same placement and dimensions')
    print('PASS valid DAE export retains multiple meshes, material groups, UV sets, vertex alpha and mesh metadata',flush=True)
    client.js('document.getElementById("mesh-overwrite").click()');client.until('!urdfWorkbench.meshSimplifier.writing&&document.getElementById("mesh-status").textContent==="已覆盖源文件"',50)
    check('(async()=>await(await daeTest.handle.getFile()).text()===daeTest.downloads[0].text&&(await(await daeTest.robot.getFile()).text())==='+json.dumps(ROBOT)+')()','overwrite writes only the original DAE and preserves URDF text')
    check('Math.abs(urdfWorkbench.viewer.robot.joints.hinge.angle-.4)<1e-7&&!document.querySelector("#status.error")','overwritten DAE refreshes visual and collision geometry without losing joint pose')
    rendered_frame(client);client.screenshot(str(output/'mesh-simplifier-dae.png'))
    client.js('document.getElementById("mesh-close").click()')
    # Exercise actual robot assets from several exporters, not only a synthetic fixture.
    for path in ['urdf/go2/dae/foot.dae','urdf/anymal_c/meshes/base.dae','urdf/Panda/meshes/visual/link0.dae']:
        result=probe(client,(ROOT/path).read_text(),True)
        assert 'error' not in result,(path,result)
        assert 0<result['after']<result['before'],(path,{k:v for k,v in result.items() if k!='text'})
        ET.fromstring(result['text'])
        print('PASS repository DAE simplification: '+path+' '+str(result['before'])+' → '+str(result['after']),flush=True)
    # Polygon triangulation, prefixed namespaces, malformed XML and unsupported skins.
    minimal='<c:COLLADA xmlns:c="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><c:library_geometries><c:geometry id="g"><c:mesh><c:source id="p"><c:float_array id="a" count="15">0 0 0 2 0 0 2 2 0 1 1 0 0 2 0</c:float_array><c:technique_common><c:accessor source="#a" count="5" stride="3"><c:param name="X" type="float"/><c:param name="Y" type="float"/><c:param name="Z" type="float"/></c:accessor></c:technique_common></c:source><c:vertices id="v"><c:input semantic="POSITION" source="#p"/></c:vertices>PRIMITIVE</c:mesh></c:geometry></c:library_geometries></c:COLLADA>'
    for primitive,faces in [('<c:polylist count="1"><c:input semantic="VERTEX" source="#v" offset="0"/><c:vcount>5</c:vcount><c:p>0 1 2 3 4</c:p></c:polylist>',3),('<c:polygons count="1"><c:input semantic="VERTEX" source="#v" offset="0"/><c:p>0 1 2 3 4</c:p></c:polygons>',3),('<c:tristrips count="1"><c:input semantic="VERTEX" source="#v" offset="0"/><c:p>0 1 4 2</c:p></c:tristrips>',2),('<c:trifans count="1"><c:input semantic="VERTEX" source="#v" offset="0"/><c:p>0 1 2 3</c:p></c:trifans>',2)]:
        result=probe(client,minimal.replace('PRIMITIVE',primitive));assert result.get('faces')==faces,result
    print('PASS prefixed DAE namespaces, concave polygons, polylist, triangle strips and fans',flush=True)
    for text,key in [('<COLLADA>','invalidDAE'),(original.replace('</COLLADA>','<library_controllers><controller id="skin"/></library_controllers></COLLADA>'),'animatedDAE'),(original.replace('<p>0 1 2</p>','<p>0 1 999999</p>'),'invalidDAE')]:
        result=probe(client,text);assert result.get('error')==key,result
    print('PASS invalid XML, out-of-range indices and animated/skinned DAE fail explicitly',flush=True)
    import_meshes(client,{'standalone/sphere.dae':original});client.until('urdfWorkbench.meshSimplifier.files.includes("standalone/sphere.dae")')
    client.js('document.getElementById("mesh-open").click()');client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    check('document.getElementById("mesh-name").textContent==="standalone/sphere.dae"','DAE-only folders also open without a robot description')
    client.js('document.getElementById("mesh-close").click()')
    client.js('document.getElementById("example-select").value="urdf/anymal_c/urdf/anymal_c.urdf";document.getElementById("load-example").click()')
    client.until('document.getElementById("filename").value==="anymal_c.urdf"&&!urdfWorkbench.loading',60)
    check('urdfWorkbench.meshSimplifier.files.includes("anymal_c/meshes/base.dae")','bundled-model file selector includes its original DAE mesh paths')
    client.js('document.getElementById("mesh-source").value="anymal_c/meshes/base.dae";document.getElementById("mesh-source").dispatchEvent(new Event("change"));document.getElementById("mesh-open").click()')
    client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    client.js('document.getElementById("mesh-run").click()')
    client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy',40)
    check('urdfWorkbench.meshSimplifier.result.faces<urdfWorkbench.meshSimplifier.original.faces','bundled ANYmal DAE simplifies through the normal UI with default settings')
    client.screenshot(str(output/'mesh-simplifier-anymal-dae.png'))
    client.js('document.getElementById("mesh-close").click()')
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    assert not [e for e in client.events if e.get('method')=='Network.requestWillBeSent' and '/api/' in e['params']['request']['url']]
    print('PASS DAE simplification runs locally with no backend or uncaught exceptions',flush=True)


if __name__=='__main__':
    smoke.run=run;smoke.main()
