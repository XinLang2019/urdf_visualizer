"""Real-browser native-source editing, save-in-place and multi-model selection checks."""
import json
from pathlib import Path
import browser_smoke as smoke
from model_interaction import rendered_frame, pointer

ROOT=Path(__file__).resolve().parents[1]
URDF='<robot name="first"><link name="base"><visual><geometry><box size=".4 .3 .2"/></geometry></visual></link></robot>'


def upload(client,files,choose=None):
    client.js('''(()=>{const d=new DataTransfer();for(const [path,text]of FILES){const f=new File([text],path.split('/').pop());Object.defineProperty(f,'webkitRelativePath',{value:path});d.items.add(f);}const input=document.getElementById('folder-input');input.files=d.files;input.dispatchEvent(new Event('change'));})()'''.replace('FILES',json.dumps(list(files.items()))))
    if choose:
        client.until('!!document.querySelector("dialog[open]")')
        client.js('[...document.querySelector("dialog[open]").querySelectorAll("button")].find(b=>b.textContent.endsWith('+json.dumps(choose)+')).click()')
    client.until('!urdfWorkbench.loading&&!!urdfWorkbench.viewer.robot')
    rendered_frame(client)


def switch(client,index):
    client.js(f'''(()=>{{document.querySelector('#model-tabs button[data-model-id="'+urdfWorkbench.models[{index}].id+'"]').click();}})()''')
    rendered_frame(client)


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    check=lambda expr,msg:smoke.check(client,expr,msg)
    upload(client,{'first.urdf':URDF})
    mjcf=(ROOT/'examples/formats/two-link.xml').read_text()
    usd=(ROOT/'examples/formats/two-link.usda').read_text()
    upload(client,{'pair/model.xml':mjcf,'pair/model.usda':usd},'model.xml')
    check('urdfWorkbench.models.length===2&&document.getElementById("source").value.startsWith("<mujoco")&&document.getElementById("filename").value==="model.xml"','MJCF editor shows original XML, separate from rendering URDF')
    client.js('window.mjcfRobot=urdfWorkbench.viewer.robot;urdfWorkbench.viewer.setJointValue("hinge",.35)')
    client.js('document.getElementById("source").value+="<!-- draft -->";document.getElementById("source").dispatchEvent(new Event("input"))')
    switch(client,0);switch(client,1)
    check('document.getElementById("source").value.endsWith("<!-- draft -->")&&Math.abs(urdfWorkbench.viewer.robot.joints.hinge.angle-.35)<1e-7','switching preserves each model draft and joint pose')
    client.js('document.getElementById("tab-files").click();document.querySelector(`#file-tree button[data-path="pair/model.usda"]`).click()')
    client.until('urdfWorkbench.models.length===3&&!urdfWorkbench.loading')
    check('document.getElementById("source").value.startsWith("#usda")&&document.getElementById("filename").value==="model.usda"&&document.getElementById("source-format").textContent.startsWith("USDA")','file tree opens native USDA source instead of generated URDF')
    client.js('document.querySelector(`#file-tree button[data-path="pair/model.xml"]`).click()')
    client.until('urdfWorkbench.viewer.robot===mjcfRobot')
    check('urdfWorkbench.models.length===3&&document.getElementById("source").value.endsWith("<!-- draft -->")','file tree reuses an existing model and restores its source draft')
    # Bind the original native source to a real disposable OPFS file, not a mock stream.
    client.js('''(async()=>{window.nativeSaveDir=await navigator.storage.getDirectory();window.nativeHandles={};for(const [name,text]of FILES){const h=await nativeSaveDir.getFileHandle(name,{create:true});const w=await h.createWritable();await w.write(text);await w.close();nativeHandles[name]=h;}window.showOpenFilePicker=async()=>[nativeHandles[document.getElementById('filename').value]];})()'''.replace('FILES',json.dumps([['model.xml',mjcf],['model.usda',usd]])))
    client.js('document.getElementById("save").click()');client.until('!document.getElementById("save").disabled')
    check('(async()=>((await (await nativeHandles["model.xml"].getFile()).text()).endsWith("<!-- draft -->")&&document.getElementById("save").textContent==="保存"))()','Save commits MJCF text to its original source file')
    client.js('document.getElementById("source").value=document.getElementById("source").value.replace("two_link_mjcf","mjcf_applied");document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot.robotName==="mjcf_applied"&&!urdfWorkbench.loading')
    check('urdfWorkbench.models.length===3&&document.getElementById("source").value.startsWith("<mujoco")','applying native edits rebuilds only that model and retains native code')
    switch(client,2)
    client.js('document.getElementById("source").value+="\\n# USD draft\\n";document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("save").click()')
    client.until('!document.getElementById("save").disabled')
    check('(async()=>((await (await nativeHandles["model.usda"].getFile()).text()).endsWith("# USD draft\\n")))()','Save commits USDA source without converting it to XML')
    client.js('window.beforeUsd=urdfWorkbench.viewer.robot;document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot!==beforeUsd&&!urdfWorkbench.loading')
    check('urdfWorkbench.models.length===3&&document.getElementById("source").value.startsWith("#usda")','USDA edits reload through the native parser')
    # A bad native edit must preserve all existing scene objects.
    client.js('window.goodUsd=document.getElementById("source").value;window.goodRobot=urdfWorkbench.viewer.robot;document.getElementById("source").value="#usda 1.0\\ndef Xform";document.getElementById("reload").click()')
    client.until('!document.getElementById("editor-error").hidden')
    check('urdfWorkbench.viewer.robot===goodRobot&&urdfWorkbench.models.length===3','invalid native code preserves the current and other models')
    client.js('document.getElementById("source").value=goodUsd;document.getElementById("source").dispatchEvent(new Event("input"));if(!document.getElementById("models-show-all").checked)document.getElementById("models-show-all").click()')
    check('urdfWorkbench.models.every(m=>m.visible)&&urdfWorkbench.models[2].offset[0]>urdfWorkbench.models[1].offset[0]','show-all renders every model with automatic non-overlapping offsets')
    client.js('document.getElementById("model-spacing-toggle").click();document.getElementById("model-reference").value=urdfWorkbench.models[0].id;document.getElementById("model-reference").dispatchEvent(new Event("change"));for(const [a,n]of [["x",2],["y",.6],["z",.4]]){const i=document.getElementById("model-offset-"+a);i.value=n;i.dispatchEvent(new Event("change"));}document.getElementById("fit-view").click()')
    check('JSON.stringify(urdfWorkbench.models[2].offset)===JSON.stringify([2,.6,.4])&&document.getElementById("model-distance").textContent.includes("2.126")','relative XYZ placement and measured origin distance use metres')
    # Large native files highlight only the visible lines while retaining complete editable text.
    client.js('window.largeNative=goodUsd+"\\n"+Array.from({length:18000},(_,i)=>"# mesh row "+i+" (1, 2, 3)").join("\\n");document.getElementById("source").value=largeNative;document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("source").scrollTop=19*17000;document.getElementById("source").dispatchEvent(new Event("scroll"))')
    check('document.getElementById("source").value===largeNative&&document.getElementById("source-highlight").childNodes.length<400&&document.getElementById("source-highlight").textContent.includes("# mesh row 16")&&document.getElementById("source-highlight").textContent.length<10000&&document.getElementById("source").scrollTop>300000','large USDA keeps full editable source with bounded visible-line highlighting')
    client.js('document.getElementById("source").value=goodUsd;document.getElementById("source").dispatchEvent(new Event("input"))')
    # Pick an inactive model through real canvas events. Same link names must stay scoped.
    switch(client,0)
    point=client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer,m=urdfWorkbench.models[2],link=m.robot.links.base,box=new T.Box3();v.scene.updateMatrixWorld(true);for(const c of link.children)if(c.isURDFVisual)box.expandByObject(c);const p=box.getCenter(new T.Vector3()).project(v.camera);return {x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2};})()''')
    client.screenshot(str(output/'multi-model-before-pick.png'))
    pointer(client,'mouseMoved',point,button='none');pointer(client,'mousePressed',point,clickCount=1);pointer(client,'mouseReleased',point,clickCount=1)
    check('urdfWorkbench.models[2].active&&document.getElementById("source").value.startsWith("#usda")&&urdfWorkbench.selectedObject===urdfWorkbench.models[2].robot.links.base','clicking a visible inactive model selects its own link and native source')
    client.js('document.querySelector("[data-display=collision]").click()')
    check('urdfWorkbench.models.every(m=>Object.values(m.robot.links).every(l=>l.children.filter(c=>c.isURDFCollider).every(c=>c.visible)))','display settings apply consistently to every loaded model')
    client.screenshot(str(output/'multi-model-desktop.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    check('document.body.scrollWidth===innerWidth&&document.getElementById("model-manager").getBoundingClientRect().right<=innerWidth','model manager fits mobile layout')
    client.screenshot(str(output/'multi-model-mobile.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    switch(client,2);client.js('document.querySelector(".model-tab.is-active .model-tab-close").click()')
    check('urdfWorkbench.models.length===2&&urdfWorkbench.models.filter(m=>m.active).length===1','removing a model keeps another model active')
    switch(client,1);client.js('document.getElementById("save").click()');client.until('!document.getElementById("save").disabled')
    client.js('document.querySelector(".model-tab.is-active .model-tab-close").click();document.querySelector(".model-tab.is-active .model-tab-close").click()')
    check('urdfWorkbench.models.length===0&&!urdfWorkbench.viewer.robot&&document.getElementById("source").value===""','removing the last model clears editor and model-specific tools')
    upload(client,{'all/one.urdf':URDF,'all/two.urdf':URDF.replace('first','second')},'加载全部模型')
    check('urdfWorkbench.models.length===2&&urdfWorkbench.models.every(m=>m.visible)','Load all imports multiple model descriptions and displays them together')
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    print('PASS no uncaught browser exceptions',flush=True)

if __name__=='__main__':
    smoke.run=run;smoke.main()
