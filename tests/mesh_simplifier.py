"""Real browser workers, STL/OBJ exports and real filesystem overwrite checks."""
import base64
import io
import json
from pathlib import Path
import struct
import browser_smoke as smoke
from model_interaction import rendered_frame

ROBOT='''<robot name="mesh_fixture"><link name="base"><visual><geometry><box size=".1 .1 .1"/></geometry></visual></link><link name="arm"><visual><geometry><mesh filename="meshes/sphere.stl" scale=".1 .1 .1"/></geometry></visual><visual><origin xyz="0 .5 0"/><geometry><mesh filename="meshes/sphere.stl" scale=".1 .1 .1"/></geometry></visual><collision><geometry><mesh filename="meshes/sphere.stl" scale=".1 .1 .1"/></geometry></collision></link><joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="2" velocity="2"/></joint></robot>'''


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1000,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    smoke.check(client,'document.getElementById("mesh-open").disabled','mesh tools are disabled until STL / OBJ / DAE files are loaded')
    client.js('''(async()=>{
      const T=parcelRequire('../node_modules/three/build/three.module.js'),g=new T.SphereGeometry(1,48,32),p=g.attributes.position,n=g.attributes.normal,uv=g.attributes.uv,idx=g.index;
      const stl=['solid mesh'],obj=['mtllib colors.mtl','o sphere'];
      for(let i=0;i<p.count;i++){obj.push('v '+[p.getX(i)+2,p.getY(i)+3,p.getZ(i)+4].join(' '));obj.push('vt '+[uv.getX(i),uv.getY(i)].join(' '));obj.push('vn '+[n.getX(i),n.getY(i),n.getZ(i)].join(' '));}
      for(let i=0;i<idx.count;i+=3){stl.push('facet normal 0 0 0','outer loop');for(let k=0;k<3;k++){const j=idx.getX(i+k);stl.push('vertex '+[p.getX(j)+2,p.getY(j)+3,p.getZ(j)+4].join(' '));}stl.push('endloop','endfacet');if(i===0)obj.push('usemtl red','s 1');if(i===Math.floor(idx.count/6)*3)obj.push('usemtl blue');obj.push('f '+[0,1,2].map(k=>{const j=idx.getX(i+k)+1;return j+'/'+j+'/'+j}).join(' '));}stl.push('endsolid mesh');g.dispose();
      window.meshTest={stl:stl.join('\\n'),obj:obj.join('\\n'),open:window.showOpenFilePicker,folder:window.showDirectoryPicker,click:HTMLAnchorElement.prototype.click,downloads:[]};
      meshTest.put=async(h,data)=>{const w=await h.createWritable();await w.write(data);await w.close()};
      meshTest.dir=await(await navigator.storage.getDirectory()).getDirectoryHandle('mesh-tests',{create:true});const meshes=await meshTest.dir.getDirectoryHandle('meshes',{create:true});
      meshTest.stlHandle=await meshes.getFileHandle('sphere.stl',{create:true});meshTest.objHandle=await meshes.getFileHandle('sphere.obj',{create:true});meshTest.wrongHandle=await meshes.getFileHandle('wrong.obj',{create:true});
      meshTest.robot=await meshTest.dir.getFileHandle('robot.urdf',{create:true});
      await meshTest.put(meshTest.stlHandle,meshTest.stl);await meshTest.put(meshTest.objHandle,meshTest.obj);await meshTest.put(meshTest.wrongHandle,meshTest.obj);
      await meshTest.put(await meshes.getFileHandle('colors.mtl',{create:true}),'newmtl red\\nKd 1 0 0\\nnewmtl blue\\nKd 0 0 1\\n');
      HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.arrayBuffer()).then(b=>{const bytes=new Uint8Array(b);let s='';for(let i=0;i<bytes.length;i+=16384)s+=String.fromCharCode(...bytes.subarray(i,i+16384));meshTest.downloads.push({name,data:btoa(s)})});}else meshTest.click.call(this)};
    })()''')
    client.js('(async()=>{await meshTest.put(meshTest.robot,'+json.dumps(ROBOT)+');window.showDirectoryPicker=async()=>meshTest.dir;document.getElementById("load-folder").click()})()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="mesh_fixture"&&!urdfWorkbench.loading',40)
    smoke.check(client,'urdfWorkbench.meshSimplifier.files.length===3','loaded folder lists STL / OBJ sources, including nested mesh folders')
    client.js('''urdfWorkbench.viewer.setJointValue("hinge",.4);document.querySelector('.file-entry[data-path$="sphere.stl"]').click()''')
    client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    smoke.check(client,'document.getElementById("mesh-dialog").open&&urdfWorkbench.meshSimplifier.original.faces>2000','file-tree mesh opens the worker-parsed comparison preview')
    client.js('document.getElementById("mesh-run").click()')
    client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy',40)
    smoke.check(client,'urdfWorkbench.meshSimplifier.result.faces<urdfWorkbench.meshSimplifier.original.faces*.8','STL simplifier reduces dense triangles')
    smoke.check(client,'(async()=>await(await meshTest.stlHandle.getFile()).text()===meshTest.stl)()','simplifying and previewing leave the original disk file intact')
    client.js('document.getElementById("mesh-download").click()');client.until('meshTest.downloads.length===1')
    item=client.js('meshTest.downloads[0]');data=base64.b64decode(item['data']);faces=struct.unpack_from('<I',data,80)[0]
    assert len(data)==84+faces*50 and faces==client.js('urdfWorkbench.meshSimplifier.result.faces')
    assert item['name']=='sphere.simplified.stl'
    coords=[struct.unpack_from('<3f',data,84+i*50+12+j*12) for i in range(faces) for j in range(3)]
    assert all(1-1e-5<=p[0]<=3+1e-5 and 2-1e-5<=p[1]<=4+1e-5 and 3-1e-5<=p[2]<=5+1e-5 for p in coords)
    print('PASS STL download is a valid compact binary mesh and retains authored coordinates',flush=True)
    client.js('meshTest.savedFaceCount=urdfWorkbench.meshSimplifier.result.faces;document.getElementById("mesh-overwrite").click()')
    client.until('!urdfWorkbench.meshSimplifier.writing&&document.getElementById("mesh-status").textContent==="已覆盖源文件"',45)
    smoke.check(client,'(async()=>{const b=await(await meshTest.stlHandle.getFile()).arrayBuffer();return new DataView(b).getUint32(80,true)===meshTest.savedFaceCount&&(await(await meshTest.robot.getFile()).text())==='+json.dumps(ROBOT)+'})()','overwrite commits only the original mesh, preserving URDF code')
    smoke.check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.hinge.angle-.4)<1e-7','refreshing saved geometry preserves the robot joint pose')
    smoke.check(client,'(()=>{const counts=[];for(const child of urdfWorkbench.viewer.robot.links.arm.children)if(child.isURDFVisual||child.isURDFCollider)child.traverse(o=>{if(o.isMesh)counts.push((o.geometry.index?.count||o.geometry.attributes.position.count)/3)});return counts.length>=3&&counts.every(n=>n===meshTest.savedFaceCount)})()','all visual and collision instances refresh from the overwritten mesh')
    rendered_frame(client);client.screenshot(str(output/'mesh-simplifier-stl.png'))
    client.js('''document.getElementById("mesh-close").click();document.querySelector('.file-entry[data-path$="sphere.obj"]').click()''')
    client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    client.js('document.getElementById("mesh-target").value=Math.floor(urdfWorkbench.meshSimplifier.original.faces*.65);document.getElementById("mesh-target").dispatchEvent(new Event("input"));document.getElementById("mesh-run").click()')
    client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy',40)
    smoke.check(client,'urdfWorkbench.meshSimplifier.result.faces<urdfWorkbench.meshSimplifier.original.faces','OBJ attribute-aware simplifier reduces triangles')
    client.js('document.getElementById("mesh-download").click()');client.until('meshTest.downloads.length===2')
    text=base64.b64decode(client.js('meshTest.downloads[1].data')).decode();lines=text.splitlines()
    assert 'mtllib colors.mtl' in lines and 'usemtl red' in lines and 'usemtl blue' in lines
    assert any(l.startswith('vt ') for l in lines) and any(l.startswith('vn ') for l in lines)
    vertex_count=sum(l.startswith('v ') for l in lines);uv_count=sum(l.startswith('vt ') for l in lines);normal_count=sum(l.startswith('vn ') for l in lines)
    for line in lines:
        if line.startswith('f '):
            for corner in line.split()[1:]:
                v,t,n=map(int,corner.split('/'));assert 1<=v<=vertex_count and 1<=t<=uv_count and 1<=n<=normal_count
    print('PASS OBJ export preserves valid position/UV/normal indices, materials and MTL reference',flush=True)
    client.js('document.getElementById("mesh-wireframe").click()');rendered_frame(client);client.screenshot(str(output/'mesh-simplifier-obj.png'))
    client.js('meshTest.changedOBJ=meshTest.obj+"\\n# external edit\\n";meshTest.put(meshTest.objHandle,meshTest.changedOBJ)')
    client.js('document.getElementById("mesh-overwrite").click()');client.until('!urdfWorkbench.meshSimplifier.writing')
    smoke.check(client,'document.querySelector("#mesh-status.error")&&document.getElementById("mesh-status").textContent.includes("变化")','externally modified mesh sources cannot be overwritten')
    smoke.check(client,'(async()=>await(await meshTest.objHandle.getFile()).text()===meshTest.changedOBJ)()','conflict keeps the externally modified file unchanged')
    client.js('document.getElementById("mesh-close").click();(()=>{const d=new DataTransfer();d.items.add(new File([meshTest.obj],"sphere.obj"));d.items.add(new File(["<metadata/>"],"notes.xml"));const i=document.getElementById("file-input");i.files=d.files;i.dispatchEvent(new Event("change"))})()')
    client.until('urdfWorkbench.meshSimplifier.files.length===1')
    client.js('document.getElementById("mesh-open").click()');client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy')
    client.js('document.getElementById("mesh-run").click()');client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy')
    client.js('window.showOpenFilePicker=async()=>[meshTest.wrongHandle];document.getElementById("mesh-overwrite").click()');client.until('!urdfWorkbench.meshSimplifier.writing')
    smoke.check(client,'document.getElementById("mesh-status").textContent.includes("同名")','file-input imports require selecting the matching source filename')
    client.js('meshTest.put(meshTest.objHandle,meshTest.obj)')
    client.js('window.showOpenFilePicker=async()=>[meshTest.objHandle];document.getElementById("mesh-overwrite").click()');client.until('!urdfWorkbench.meshSimplifier.writing&&document.getElementById("mesh-status").textContent==="已覆盖源文件"')
    smoke.check(client,'(async()=>{const text=await(await meshTest.objHandle.getFile()).text();return text.includes("mtllib colors.mtl")&&text.split("\\n").filter(l=>l.startsWith("f ")).length===urdfWorkbench.meshSimplifier.result.faces})()','standalone OBJ import can bind and overwrite the real original source')
    client.js('document.getElementById("mesh-close").click();document.getElementById("language").click()')
    smoke.check(client,'document.getElementById("mesh-open").textContent==="Simplify…"','mesh tools follow the workbench language')
    client.js('document.getElementById("mesh-open").click()');client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy')
    for w,h in [(390,844),(1280,900)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':w,'height':h,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        smoke.check(client,'document.getElementById("mesh-dialog").scrollWidth<=document.getElementById("mesh-dialog").clientWidth','mesh dialog fits '+str(w)+'px viewport')
    client.js('document.getElementById("mesh-run").click();document.getElementById("mesh-close").click()')
    smoke.check(client,'!document.getElementById("mesh-dialog").open&&!urdfWorkbench.meshSimplifier.busy','closing cancels the worker and clears pending results')
    client.js('document.getElementById("example-select").value="urdf/cassie/urdf/cassie.urdf";document.getElementById("load-example").click()')
    client.until('document.getElementById("filename").value==="cassie.urdf"&&!urdfWorkbench.loading',60)
    smoke.check(client,'urdfWorkbench.meshSimplifier.files.includes("cassie/meshes/torso.stl")','bundled robot catalog exposes its actual STL meshes')
    client.js('document.getElementById("mesh-source").value="cassie/meshes/torso.stl";document.getElementById("mesh-source").dispatchEvent(new Event("change"));document.getElementById("mesh-open").click()')
    client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy',40)
    client.js('document.getElementById("mesh-run").click()');client.until('!!urdfWorkbench.meshSimplifier.result&&!urdfWorkbench.meshSimplifier.busy',40)
    smoke.check(client,'urdfWorkbench.meshSimplifier.result.faces<urdfWorkbench.meshSimplifier.original.faces','real Cassie torso STL is simplified successfully')
    client.js('''(async()=>{const file=await(await fetch('urdf/cassie/meshes/torso.stl')).blob();meshTest.catalogHandle=await meshTest.dir.getFileHandle('torso.stl',{create:true});await meshTest.put(meshTest.catalogHandle,file);window.showOpenFilePicker=async()=>[meshTest.catalogHandle];document.getElementById('mesh-overwrite').click()})()''')
    client.until('!urdfWorkbench.meshSimplifier.writing&&document.getElementById("mesh-status").textContent==="Source overwritten"',60)
    smoke.check(client,'!document.querySelector("#status.error")','bundled-model mesh override refreshes references without losing other model assets')
    client.screenshot(str(output/'mesh-simplifier-cassie.png'))
    client.js('document.getElementById("mesh-close").click()')
    requests=[e['params']['request']['url'] for e in client.events if e.get('method')=='Network.requestWillBeSent']
    assert not any('/api/' in u for u in requests)
    exceptions=[e for e in client.events if e.get('method')=='Runtime.exceptionThrown'];assert not exceptions,exceptions
    print('PASS mesh simplification runs locally with no backend, uploads or uncaught exceptions',flush=True)


if __name__=='__main__':
    smoke.run=run;smoke.main()
