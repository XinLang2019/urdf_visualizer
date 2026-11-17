/* Mesh Simplifier UI, comparison preview and explicit source-file commits. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),supported=path=>/\.(stl|obj)$/i.test(path);
  const words={
    choose:['选择 STL / OBJ 网格…','Choose an STL / OBJ mesh…'],empty:['未加载 STL / OBJ','No STL / OBJ loaded'],open:['简化…','Simplify…'],
    original:['原始','Original'],result:['简化后','Simplified'],retain:['保留面数','Keep faces'],target:['目标面数','Target faces'],error:['最大误差 (%)','Max. error (%)'],
    border:['保留边界','Lock borders'],wireframe:['线框','Wireframe'],run:['开始简化','Simplify'],overwrite:['覆盖源文件','Overwrite source'],download:['下载','Download'],
    reading:['读取中…','Reading…'],parsing:['解析中…','Parsing…'],simplifying:['简化中…','Simplifying…'],exporting:['导出中…','Exporting…'],
    ready:['可设置目标面数','Set the target face count'],done:['简化完成','Simplified'],limited:['受形状保护，未达到目标面数','Shape protection prevented reaching the target'],unchanged:['未减少面数，可调高误差或解除边界保护','No reduction; increase error or unlock borders'],
    saved:['已覆盖源文件','Source overwritten'],downloaded:['已下载','Downloaded'],writing:['保存中…','Saving…'],cancelled:['已取消','Cancelled'],failed:['操作失败','Operation failed'],
    chooseSource:['选择对应的原始网格文件','Select the original mesh file'],wrongSource:['请选择同名原文件','Select the matching original filename'],changed:['源文件内容已变化或选错文件，请重新加载后简化','The source changed or the wrong file was selected. Reload it before simplifying.'],
    unsupportedSave:['浏览器不支持覆盖源文件，请使用 Chrome / Edge 或下载','Use Chrome / Edge to overwrite, or download the result'],denied:['未获得源文件写入权限','Source write permission was not granted'],
    workerFailed:['无法启动简化器，请通过 HTTP(S) 打开页面','Could not start the simplifier. Open this page over HTTP(S).'],
    invalidNumbers:['模型包含无效数值','The mesh contains invalid numbers'],invalidSTL:['STL 数据不完整','Invalid STL data'],invalidOBJIndex:['OBJ 顶点索引无效','Invalid OBJ vertex index'],
    emptyMesh:['模型没有可简化的三角面','The mesh contains no triangles'],emptyResult:['无法生成有效简化结果','Could not produce a valid simplified mesh'],tooLarge:['模型超过 384 MiB 或 800 万面','The mesh exceeds 384 MiB or 8 million faces'],
    invalidSettings:['请设置有效的面数和误差','Enter a valid face count and error'],complexPolygon:['请先将超过 512 个顶点的多边形三角化','Triangulate polygons with more than 512 vertices first'],invalidPolygon:['多边形无法三角化','The polygon cannot be triangulated'],
    mixedAttributes:['OBJ 面的 UV / 法线数据不完整','Incomplete OBJ face UV / normal data'],notTriangles:['仅支持网格，不支持 OBJ 曲线或点云','Only meshes are supported; OBJ curves and point clouds are not supported'],
    weightedOBJ:['请先烘焙 OBJ 齐次坐标','Bake OBJ homogeneous coordinates first'],volumeUV:['暂不支持三维纹理坐标','3D texture coordinates are not supported'],wasmUnsupported:['浏览器不支持 WebAssembly','WebAssembly is unavailable'],
    faces:['面','faces'],vertices:['顶点','vertices'],mtl:['OBJ 下载后请与原 MTL / 贴图放在一起','Keep the downloaded OBJ beside its original MTL / textures'],
    loaded:['已加载网格','Meshes loaded'],updateFailed:['文件已保存，视图刷新失败，请重新加载模型','File saved; refreshing the view failed. Reload the model.']
  };
  const digest=async file=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),v=>v.toString(16).padStart(2,'0')).join('');
  window.URDFMeshSimplifier=class {
    constructor({THREE,viewer,getLanguage,onSaved}){
      this.T=THREE;this.viewer=viewer;this.getLanguage=getLanguage;this.onSaved=onSaved;this.entries=new Map();this.context=null;this.generation=0;this.busy=false;this.writing=false;this.messageKey='';this.messageError=false;
      $('mesh-open').onclick=()=>this.open($('mesh-source').value);$('mesh-source').onchange=()=>{$('mesh-source').title=$('mesh-source').value;$('mesh-open').disabled=!$('mesh-source').value;};
      $('mesh-close').onclick=()=>this.close();$('mesh-dialog').addEventListener('cancel',e=>{e.preventDefault();this.close();});
      $('mesh-run').onclick=()=>this.run();$('mesh-download').onclick=()=>this.download();$('mesh-overwrite').onclick=()=>this.overwrite();
      $('mesh-ratio').oninput=()=>{if(this.original)$('mesh-target').value=Math.max(1,Math.floor(this.original.faces*Number($('mesh-ratio').value)/100));this.invalidate();this.updateRatio();};
      $('mesh-target').oninput=()=>{if(this.original)$('mesh-ratio').value=Math.max(1,Math.min(99,100*Number($('mesh-target').value)/this.original.faces));this.invalidate();this.updateRatio();};
      for(const id of ['mesh-error','mesh-border'])$(id).onchange=()=>this.invalidate();
      $('mesh-wireframe').onchange=()=>{for(const object of [this.originalGroup,this.resultGroup])object?.traverse(o=>{if(o.material)o.material.wireframe=$('mesh-wireframe').checked;});};
      this.translate();this.setContext(null);
    }
    t(key){return words[key]?.[this.getLanguage()==='zh'?0:1]||key;}
    translate(){
      for(const el of document.querySelectorAll('[data-mesh-i18n]'))el.textContent=this.t(el.dataset.meshI18n);
      $('mesh-source').setAttribute('aria-label',this.t('choose'));$('mesh-target').setAttribute('aria-label',this.t('target'));$('mesh-error').setAttribute('aria-label',this.t('error'));$('mesh-close').setAttribute('aria-label',this.getLanguage()==='zh'?'关闭网格简化':'Close Mesh Simplifier');
      this.refreshOptions();this.updateCounts();if(this.messageKey)this.setMessage(this.messageKey,this.messageError);this.updateRatio();
    }
    setContext(context){
      if(context!==this.context){this.close();this.context=context;}
      this.entries=new Map();
      if(context?.assets){for(const [path,asset] of context.assets)if(supported(path)&&!path.split('/').includes('__converted__'))this.entries.set(path,{path,asset});}
      else if(context?.files){for(const path of context.files)if(supported(path))this.entries.set(path,{path,asset:context.meshOverrides?.get(path)||null});}
      this.refreshOptions();
    }
    refreshOptions(){
      const select=$('mesh-source'),selected=select.value;select.replaceChildren(new Option(this.entries.size?this.t('choose'):this.t('empty'),''));for(const path of [...this.entries.keys()].sort())select.add(new Option(path,path));if(this.entries.has(selected))select.value=selected;else if(this.entries.size===1)select.value=[...this.entries.keys()][0];
      select.disabled=!this.entries.size;$('mesh-open').disabled=!select.value;$('mesh-source').title=select.value;
    }
    async asset(entry){
      if(!entry.asset){const url=new URL('urdf/'+entry.path,location.href),r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status);entry.asset={name:entry.path.split('/').pop(),file:new File([await r.blob()],entry.path.split('/').pop()),handle:null,url:null,savedText:null};}
      const asset=entry.asset;if(asset.handle)asset.file=await asset.handle.getFile();return asset;
    }
    async open(path){
      const entry=this.entries.get(path);if(!entry)return;$('mesh-source').value=path;$('mesh-open').disabled=false;this.close();const generation=++this.generation;this.entry=entry;this.original=null;this.result=null;this.busy=true;this.selectedContext=this.context;
      $('mesh-name').textContent=path;$('mesh-name').title=path;$('mesh-ratio').value=50;$('mesh-error').value=1;$('mesh-progress').hidden=false;$('mesh-progress').value=0;$('mesh-dialog').showModal();this.setMessage('reading');this.buttons();this.updateCounts();
      try{
        const asset=await this.asset(entry);const baselineHash=await digest(asset.file);if(generation!==this.generation)return;this.source=asset;this.baselineHash=baselineHash;
        this.initPreview();this.disposeGroups();const workerPath=document.querySelector('script[data-mesh-worker]')?.dataset.meshWorker||'mesh-simplifier-worker.js';this.worker=new Worker(new URL(workerPath,document.baseURI),{type:'module'});
        this.worker.onerror=e=>{e.preventDefault();if(generation===this.generation){this.busy=false;$('mesh-progress').hidden=true;this.setMessage('workerFailed',true);this.buttons();}};
        this.worker.onmessage=({data})=>{if(generation!==this.generation)return;try{this.receive(data);}catch(e){this.busy=false;this.setMessage(e.message,true);this.buttons();}};
        this.worker.postMessage({action:'load',file:asset.file});
      }catch(e){if(generation===this.generation){this.busy=false;this.setMessage(e.message,true);this.buttons();}}
    }
    receive(data){
      if(data.type==='progress'){$('mesh-progress').value=data.value;this.setMessage(data.stage);return;}
      this.busy=false;$('mesh-progress').hidden=true;
      if(data.type==='error')this.setMessage(data.message,true);
      else if(data.type==='loaded'){
        this.original={...data.stats,bytes:data.bytes,libraries:data.libraries};this.originalGroup=this.group(data.parts);this.scene.add(this.originalGroup);this.fitPreview();$('mesh-target').max=Math.max(1,this.original.faces-1);$('mesh-target').value=Math.max(1,Math.floor(this.original.faces/2));this.setMessage('ready');
      }else if(data.type==='simplified'){
        this.result={...data.stats,blob:data.blob,error:data.error};if(this.resultGroup){this.scene.remove(this.resultGroup);this.dispose(this.resultGroup);}this.resultGroup=this.group(data.parts);this.resultGroup.position.copy(this.originalGroup.position);this.scene.add(this.resultGroup);
        this.setMessage(this.result.faces===this.original.faces?'unchanged':this.result.faces>Number($('mesh-target').value)?'limited':'done');
      }this.updateCounts();this.updateRatio();this.buttons();
    }
    run(){
      if(this.busy||!this.worker||!this.original)return;const target=Number($('mesh-target').value),error=Number($('mesh-error').value)/100;if(!Number.isInteger(target)||target<1||target>=this.original.faces||!Number.isFinite(error)||error<0||error>1)return this.setMessage('invalidSettings',true);
      this.invalidate();this.busy=true;this.setMessage('simplifying');$('mesh-progress').value=.2;$('mesh-progress').hidden=false;this.buttons();this.worker.postMessage({action:'simplify',ratio:target/this.original.faces,error,lockBorder:$('mesh-border').checked});
    }
    invalidate(){if(this.busy||this.writing)return;this.result=null;if(this.resultGroup){this.scene.remove(this.resultGroup);this.dispose(this.resultGroup);this.resultGroup=null;}this.updateCounts();this.buttons();if(this.original)this.setMessage('ready');}
    buttons(){const blocked=this.busy||this.writing;for(const id of ['mesh-ratio','mesh-target','mesh-error','mesh-border'])$(id).disabled=blocked||!this.original;$('mesh-run').disabled=blocked||!this.original||this.original.faces<2;$('mesh-download').disabled=blocked||!this.result;$('mesh-overwrite').disabled=blocked||!this.result;$('mesh-close').disabled=this.writing;}
    updateRatio(){$('mesh-ratio-value').textContent=Math.round(Number($('mesh-ratio').value))+'%';}
    count(info){return info?`${info.faces.toLocaleString()} ${this.t('faces')} · ${info.vertices.toLocaleString()} ${this.t('vertices')} · ${this.bytes(info.bytes??info.blob?.size??0)}`:'—';}
    bytes(n){return n>=1048576?(n/1048576).toFixed(2)+' MB':(n/1024).toFixed(1)+' KB';}
    updateCounts(){$('mesh-before-count').textContent=this.count(this.original);$('mesh-after-count').textContent=this.count(this.result);$('mesh-reduction').textContent=this.original&&this.result?`−${((1-this.result.faces/this.original.faces)*100).toFixed(1)}%`:'';}
    setMessage(key,error=false){this.messageKey=key;this.messageError=error;$('mesh-status').textContent=this.t(key);$('mesh-status').classList.toggle('error',error);$('mesh-note').textContent=this.result&&this.original?.libraries?.length?this.t('mtl'):'';}
    download(){if(!this.result||this.busy)return;const a=document.createElement('a'),url=URL.createObjectURL(this.result.blob);a.href=url;a.download=this.source.name.replace(/(\.(?:stl|obj))$/i,'.simplified$1');document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);this.setMessage('downloaded');}
    async overwrite(){
      if(!this.result||this.busy||this.writing)return;const source=this.source,result=this.result,baselineHash=this.baselineHash,context=this.selectedContext,path=this.entry.path;let handle=source.handle;
      this.writing=true;this.buttons();
      try{
        if(!handle){if(!window.showOpenFilePicker)throw Error('unsupportedSave');const extension=source.name.split('.').pop().toLowerCase();[handle]=await showOpenFilePicker({multiple:false,types:[{description:this.t('chooseSource'),accept:{'application/octet-stream':['.'+extension]}}]});if(handle.name!==source.name)throw Error('wrongSource');}
        if(handle.queryPermission&&await handle.queryPermission({mode:'readwrite'})!=='granted'&&await handle.requestPermission({mode:'readwrite'})!=='granted')throw Error('denied');
        this.setMessage('writing');const disk=await handle.getFile();if(await digest(disk)!==baselineHash)throw Error('changed');const stream=await handle.createWritable();try{await stream.write(result.blob);await stream.close();}catch(e){try{await stream.abort();}catch{}throw e;}
        source.handle=handle;source.file=new File([result.blob],source.name,{type:result.blob.type});this.baselineHash=await digest(source.file);if(source.url){URL.revokeObjectURL(source.url);source.url=null;}
        if(!context.assets){context.meshOverrides||=new Map();context.meshOverrides.set(path,source);}
        try{await this.onSaved({path,source,context});this.setMessage('saved');}catch{this.setMessage('updateFailed',true);}
      }catch(e){this.setMessage(e.name==='AbortError'?'cancelled':e.message,true);}
      finally{this.writing=false;this.buttons();}
    }
    close(){if(this.writing)return;this.generation++;this.worker?.terminate();this.worker=null;this.busy=false;this.result=null;this.original=null;this.entry=null;this.source=null;this.messageKey='';if($('mesh-dialog').open)$('mesh-dialog').close();cancelAnimationFrame(this.frame);this.disposeGroups();this.buttons();}
    dispose(object){object?.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
    disposeGroups(){for(const key of ['originalGroup','resultGroup'])if(this[key]){this.scene?.remove(this[key]);this.dispose(this[key]);this[key]=null;}}
    initPreview(){
      if(!this.renderer){const T=this.T,canvas=$('mesh-canvas');this.renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.scene=new T.Scene();this.scene.background=new T.Color(0x173b4b);this.scene.add(new T.HemisphereLight(0xffffff,0x4c7486,1));const light=new T.DirectionalLight(0xffffff,.9);light.position.set(3,5,4);this.scene.add(light);this.camera=new T.PerspectiveCamera(40,1,.001,1e6);this.controls=new this.viewer.controls.constructor(this.camera,canvas);this.controls.enableDamping=true;
        this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe($('mesh-preview'));}
      this.resize();cancelAnimationFrame(this.frame);const render=()=>{if(!$('mesh-dialog').open)return;this.frame=requestAnimationFrame(render);this.controls.update();const w=this.previewWidth,h=this.previewHeight,mid=Math.floor(w/2),r=this.renderer;r.setScissorTest(true);r.setViewport(0,0,mid,h);r.setScissor(0,0,mid,h);if(this.originalGroup)this.originalGroup.visible=true;if(this.resultGroup)this.resultGroup.visible=false;r.render(this.scene,this.camera);r.setViewport(mid,0,w-mid,h);r.setScissor(mid,0,w-mid,h);if(this.originalGroup)this.originalGroup.visible=!this.resultGroup;if(this.resultGroup)this.resultGroup.visible=true;r.render(this.scene,this.camera);};render();
    }
    resize(){if(!this.renderer)return;const r=$('mesh-preview').getBoundingClientRect();this.previewWidth=Math.max(2,Math.floor(r.width));this.previewHeight=Math.max(2,Math.floor(r.height));this.renderer.setSize(this.previewWidth,this.previewHeight,false);this.camera.aspect=this.previewWidth/2/this.previewHeight;this.camera.updateProjectionMatrix();}
    group(parts){const T=this.T,group=new T.Group();for(const p of parts){const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(p.positions,3));g.setIndex(new T.BufferAttribute(p.indices,1));if(p.normals.length)g.setAttribute('normal',new T.BufferAttribute(p.normals,3));else g.computeVertexNormals();if(p.uvs.length)g.setAttribute('uv',new T.BufferAttribute(p.uvs,2));if(p.colors.length)g.setAttribute('color',new T.BufferAttribute(p.colors,3));const material=new T.MeshPhongMaterial({color:p.colors.length?0xffffff:0xb8d4e8,vertexColors:!!p.colors.length,side:T.DoubleSide,shininess:35,wireframe:$('mesh-wireframe').checked});group.add(new T.Mesh(g,material));}return group;}
    fitPreview(){const T=this.T,box=new T.Box3().setFromObject(this.originalGroup),center=box.getCenter(new T.Vector3()),radius=Math.max(box.getSize(new T.Vector3()).length()/2,1e-6);this.originalGroup.position.copy(center).negate();this.camera.near=radius/1000;this.camera.far=radius*1000;const distance=radius/Math.sin(this.camera.fov*Math.PI/360)/Math.min(1,this.camera.aspect);this.camera.position.set(.65,.45,1).normalize().multiplyScalar(distance*1.08);this.controls.target.set(0,0,0);this.camera.updateProjectionMatrix();this.controls.update();}
    get snapshot(){return {files:[...this.entries.keys()],busy:this.busy,writing:this.writing,original:this.original?{faces:this.original.faces,vertices:this.original.vertices}:null,result:this.result?{faces:this.result.faces,vertices:this.result.vertices,bytes:this.result.blob.size}:null};}
  };
})();
