"""Model tabs, cached joint controls and high-face-count wheel switching in real Edge."""
import json,math,struct,base64,statistics
from model_interaction import rendered_frame,pointer
from multi_model_browser import upload,switch
import browser_smoke as smoke

def fixture():
    count=4000;tri=[]
    for i in range(count):
        a=i*math.tau/count;b=(i+1)*math.tau/count
        p=(.2*math.cos(a),-.04,.2*math.sin(a));q=(.2*math.cos(b),-.04,.2*math.sin(b));r=(p[0],.04,p[2]);s=(q[0],.04,q[2])
        tri.extend([(p,q,r),(q,s,r),((0,-.04,0),q,p),((0,.04,0),r,s)])
    stl=b'wheel benchmark'.ljust(80,b'\0')+struct.pack('<I',len(tri))+b''.join(struct.pack('<12fH',0,0,0,*a,*b,*c,0) for a,b,c in tri)
    xml='<robot name="wheel_benchmark"><link name="base"><inertial><mass value="10"/><inertia ixx="1" iyy="1" izz="1"/></inertial><visual><geometry><box size=".7 .4 .2"/></geometry></visual><collision><geometry><box size=".7 .4 .2"/></geometry></collision></link>'
    for n,(x,y) in enumerate([(-.4,-.3),(-.4,.3),(.4,-.3),(.4,.3)]):
        xml+=f'<link name="wheel{n}"><inertial><mass value="1"/><inertia ixx=".02" iyy=".03" izz=".02"/></inertial><visual><geometry><mesh filename="wheel.stl"/></geometry></visual><collision><geometry><mesh filename="wheel.stl"/></geometry></collision></link><joint name="wheel_joint{n}" type="continuous"><parent link="base"/><child link="wheel{n}"/><origin xyz="{x} {y} -.3"/><axis xyz="0 1 0"/></joint>'
    return xml+'</robot>',stl

def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    xml,stl=fixture()
    client.js('''(()=>{const d=new DataTransfer();d.items.add(new File([XML],'a.urdf'));d.items.add(new File([XML.replace('wheel_benchmark','wheel_other')],'b.urdf'));d.items.add(new File([Uint8Array.from(atob(DATA),c=>c.charCodeAt(0))],'wheel.stl'));const input=document.getElementById('file-input');input.files=d.files;input.dispatchEvent(new Event('change'));})()'''.replace('XML',json.dumps(xml)).replace('DATA',json.dumps(base64.b64encode(stl).decode())))
    client.until('!!document.querySelector("dialog[open]")');client.js('[...document.querySelector("dialog[open]").querySelectorAll("button")].find(b=>b.textContent==="加载全部模型").click()')
    client.until('urdfWorkbench.models.length===2&&!urdfWorkbench.loading',90)
    client.js('document.getElementById("tab-joints").click();window.prepareProbe={count:0,time:0,fn:URDFStaticContacts.prototype.prepare};URDFStaticContacts.prototype.prepare=function(...args){const t=performance.now();try{return prepareProbe.fn.apply(this,args)}finally{prepareProbe.count++;prepareProbe.time+=performance.now()-t}}')
    smoke.check(client,'document.getElementById("models-show-all").checked&&urdfWorkbench.models.every(m=>m.visible)', 'new models appear side by side without an extra action')
    client.js('window.cachedJointRows=urdfWorkbench.models.map(m=>null);window.cachedCameras=urdfWorkbench.viewer.camera.position.toArray();window.initialGeometries=urdfWorkbench.viewer.renderer.info.memory.geometries')
    rendered_frame(client)
    client.js('initialGeometries=urdfWorkbench.viewer.renderer.info.memory.geometries;cachedCameras=urdfWorkbench.viewer.camera.position.toArray()')
    records=[]
    for i in range(8):
        idx=i%2
        records.append(client.js('''(()=>{const start=performance.now(),id=urdfWorkbench.models[IDX].id,tab=document.querySelector('#model-tabs button[data-model-id="'+id+'"]');tab.click();return performance.now()-start})()'''.replace('IDX',str(idx))))
        rendered_frame(client)
        row_ok=client.js('(()=>{const index='+str(idx)+',row=document.querySelector("#joint-list li[data-name=wheel_joint0]");if(!cachedJointRows[index])cachedJointRows[index]=row;return cachedJointRows[index]===row})()')
        assert row_ok,'model switches replaced cached joint controls'
    probe=client.js('({count:prepareProbe.count,time:prepareProbe.time})')
    assert probe['count']==0,probe
    smoke.check(client,'urdfWorkbench.viewer.camera.position.toArray().every((v,i)=>Math.abs(v-cachedCameras[i])<1e-9)&&urdfWorkbench.viewer.renderer.info.memory.geometries===initialGeometries','switches retain camera and geometry resources')
    print('PASS high-face-count wheel switch times (ms): '+json.dumps(records)+'; median '+str(round(statistics.median(records),2)),flush=True)
    # The same joint name on two models must refer to separate physical joints.
    client.js('urdfWorkbench.viewer.setJointValue("wheel_joint0",.37)');switch(client,0)
    smoke.check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.wheel_joint0.angle)<1e-8&&Math.abs(urdfWorkbench.models[1].robot.joints.wheel_joint0.angle-.37)<1e-8','joint edits remain scoped to their model')
    # Enable analysis on each model once, then verify wheel fitting is cached.
    client.js('document.getElementById("statics-toggle").click()');client.until('urdfWorkbench.statics.valid')
    switch(client,1);client.until('urdfWorkbench.statics.valid')
    before=client.js('prepareProbe.count');assert before>0
    for idx in [0,1,0]:
        switch(client,idx);client.until('urdfWorkbench.statics.valid')
    smoke.check(client,'prepareProbe.count==='+str(before),'enabled analysis reuses each model wheel geometry while updating live torques')
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
    # The first press on an inactive robot must activate and drag its own joint immediately.
    point=client.js("""(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer,meshes=[];v.scene.updateMatrixWorld(true);v.camera.updateMatrixWorld(true);for(const m of urdfWorkbench.models)m.robot.traverseVisible(o=>{if(!o.isMesh)return;let p=o.parent;while(p&&!p.isURDFVisual&&!p.isURDFCollider)p=p.parent;if(p?.isURDFVisual)meshes.push(o)});const target=urdfWorkbench.models[1].robot;for(let i=0;i<4;i++){const link=target.links['wheel'+i],box=new T.Box3();for(const c of link.children)if(c.isURDFVisual)box.expandByObject(c);const center=box.getCenter(new T.Vector3()),size=box.getSize(new T.Vector3());for(const xyz of [[0,0,0],[.15,0,0],[-.15,0,0],[0,0,.15]]){const p=center.clone().add(new T.Vector3(...xyz).multiply(size)).project(v.camera),r=new T.Raycaster();r.setFromCamera(new T.Vector2(p.x,p.y),v.camera);let owner=r.intersectObjects(meshes)[0]?.object;while(owner&&!owner.isURDFLink)owner=owner.parent;if(owner===link){window.dragJointName='wheel_joint'+i;window.oldOtherJoint=urdfWorkbench.models[0].robot.joints[dragJointName].angle;window.oldDragAngle=target.joints[dragJointName].angle;return {x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2}}}}throw Error('No visible wheel for inactive-model drag')})()""")
    pointer(client,'mouseMoved',point,button='none');pointer(client,'mousePressed',point,clickCount=1)
    smoke.check(client,'urdfWorkbench.models[1].active&&urdfWorkbench.viewer.dragControls.manipulating===urdfWorkbench.models[1].robot.joints[dragJointName]','first press on another robot immediately grabs its own joint')
    moved={'x':point['x']+35,'y':point['y']+20};pointer(client,'mouseMoved',moved,buttons=1);pointer(client,'mouseReleased',moved);rendered_frame(client)
    smoke.check(client,'Math.abs(urdfWorkbench.models[1].robot.joints[dragJointName].angle-oldDragAngle)>.001&&urdfWorkbench.models[0].robot.joints[dragJointName].angle===oldOtherJoint','cross-model dragging updates only the grabbed robot joint')
    # Reuse the complete native-source test upload helper for a large tab strip.
    simple='<robot name="tabs"><link name="base"><visual><geometry><box size=".2 .2 .2"/></geometry></visual></link></robot>'
    upload(client,{'extra/model-'+str(i)+'.urdf':simple.replace('tabs','tabs'+str(i)) for i in range(5)},'加载全部模型')
    smoke.check(client,'urdfWorkbench.models.length===7&&document.querySelectorAll("#model-tabs [role=tab]").length===7','tab strip provides a separate selectable tab and close button per model')
    rendered_frame(client)
    switch(client,0);switch(client,6)
    smoke.check(client,'(()=>{const list=document.getElementById("model-tabs"),tab=document.querySelector(".model-tab.is-active"),l=list.getBoundingClientRect(),r=tab.getBoundingClientRect();return list.scrollWidth>list.clientWidth&&r.left>=l.left-1&&r.right<=l.right+1})()','overflowing model tabs scroll the active model into view')
    client.js('document.querySelector("#model-tabs [aria-selected=true]").dispatchEvent(new KeyboardEvent("keydown",{key:"Home",bubbles:true}))')
    smoke.check(client,'urdfWorkbench.models[0].active','Home selects the first model without changing joint data')
    client.js('document.querySelector("#model-tabs [aria-selected=true]").dispatchEvent(new KeyboardEvent("keydown",{key:"End",bubbles:true}))')
    smoke.check(client,'urdfWorkbench.models[6].active','End selects the last model')
    client.js('document.querySelector(".model-tab:not(.is-active) .model-tab-close").click()')
    smoke.check(client,'urdfWorkbench.models.length===6&&urdfWorkbench.models[5].active','closing an inactive tab leaves the current model selected')
    for width,height in [(1897,942),(390,844)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        smoke.check(client,'document.body.scrollWidth===innerWidth&&document.getElementById("model-manager").scrollWidth<=document.getElementById("model-manager").clientWidth','model tabs stay within the '+str(width)+'px workspace')
        client.screenshot(str(output/('model-tabs-'+str(width)+'.png')))
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    print('PASS no uncaught exceptions while switching model tabs',flush=True)
    client.screenshot(str(output/'wheel-models.png'))

if __name__=='__main__':
    smoke.run=run;smoke.main()
