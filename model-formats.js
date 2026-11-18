/* Native format bridge. URDF viewing remains usable on ordinary static hosting. */
(() => {
  'use strict';
  let serial=0;
  const generated=path=>path.split('/').includes('__converted__');
  const extension=name=>name.split('.').pop().toLowerCase();
  const normalize=path=>{const parts=[];for(const p of path.replace(/\\/g,'/').split('/')){if(p==='..')parts.pop();else if(p&&p!=='.')parts.push(p);}return parts.join('/');};
  window.URDFModelFormats=class {
    constructor(message){this.message=message;}
    static isModel(name){return /\.(urdf|xml|mjcf|usd|usda|usdc|usdz)$/i.test(name);}
    static async detect(file){
      if(['usd','usda','usdc','usdz'].includes(extension(file.name)))return 'usd';
      const xml=new DOMParser().parseFromString(await file.text(),'application/xml');
      if(xml.documentElement.tagName==='robot')return 'urdf';
      if(xml.documentElement.tagName==='mujoco')return 'mjcf';
      if(xml.documentElement.tagName==='mujocoinclude')return null;
      return extension(file.name)==='urdf'?'urdf':extension(file.name)==='mjcf'?'mjcf':null;
    }
    async request(action,entry,files,target,sourceInfo){
      // Static hosting (including GitHub Pages) always uses local browser readers.
      // The native service remains available only when explicitly requested.
      if(target==='step'||new URLSearchParams(location.search).get('formats')!=='native')return URDFBrowserFormats.process(action,entry,files,target,sourceInfo);
      if(location.protocol==='file:')throw Error(this.message('formatsServiceRequired'));
      const form=new FormData(),paths=[];
      for(const [path,file] of files){paths.push(path);form.append('files',file,file.name||path.split('/').pop());}
      form.append('paths',JSON.stringify(paths));form.append('entry',entry);if(target)form.append('target',target);if(sourceInfo)form.append('sourceInfo',JSON.stringify(sourceInfo));
      const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),200000);
      try{
        let response;
        try{response=await fetch('/api/formats/'+action,{method:'POST',body:form,signal:controller.signal});}
        catch(error){if(error.name==='AbortError')throw Error(this.message('formatsTimeout'));throw Error(this.message('formatsServiceRequired'));}
        if(!response.ok){let detail;try{detail=(await response.json()).error;}catch{}throw Error(detail||this.message('formatsServiceRequired'));}
        if(action==='import'&&!response.headers.get('Content-Type')?.includes('application/json'))throw Error(this.message('formatsServiceRequired'));
        if(action==='convert'&&!response.headers.get('Content-Type')?.includes('application/zip'))throw Error(this.message('formatsServiceRequired'));
        return response;
      }finally{clearTimeout(timer);}
    }
    async prepare(file,context){
      const format=await URDFModelFormats.detect(file);
      if(format==='urdf')return {text:await file.text(),context};
      if(!format)throw Error(this.message('unsupportedModel'));
      const files=new Map([...context.assets].filter(([path,asset])=>asset.file&&!generated(path)).map(([path,asset])=>[path,asset.file]));
      const response=await this.request('import',context.path,files),result=await response.json();
      if(typeof result.urdf!=='string'||!Array.isArray(result.assets))throw Error('Invalid format service response');
      const assets=new Map([...context.assets].filter(([p])=>!generated(p)).map(([p,a])=>[p,{...a,url:null}]));
      const prefix='__converted__/'+(++serial)+'/',folder=context.path.slice(0,context.path.lastIndexOf('/')+1);
      for(const asset of result.assets){
        const bytes=Uint8Array.from(atob(asset.data),c=>c.charCodeAt(0));
        assets.set(folder+prefix+asset.path,{name:asset.path.split('/').pop(),file:new File([bytes],asset.path.split('/').pop()),url:null});
      }
      const document=new DOMParser().parseFromString(result.urdf,'application/xml');
      for(const mesh of document.querySelectorAll('mesh[filename]'))mesh.setAttribute('filename',prefix+mesh.getAttribute('filename'));
      const text=new XMLSerializer().serializeToString(document),name=file.name.replace(/\.[^.]+$/,'')+'.viewer.urdf';
      return {text,context:{...context,name,assets,originalFormat:format,originalName:file.name,formatReport:result.report,sourceFile:{name,handle:null,savedText:null,written:false}}};
    }
    async bundle(text,context){
      let entry=normalize(context?.path||context?.name||'model.urdf');
      if(context?.originalFormat||!/\.(urdf|xml)$/i.test(entry))entry=normalize(entry.slice(0,entry.lastIndexOf('/')+1)+(context?.name||'model.viewer.urdf'));
      const files=new Map();
      if(context?.assets){for(const [path,asset] of context.assets)if(asset.file)files.set(path,asset.file);}
      else if(context?.files){
        const wanted=context.files.filter(p=>/\.(stl|dae|obj|mtl|gltf|glb|bin|png|jpg|jpeg|webp|bmp|tga)$/i.test(p));
        // Fetch a bounded number at once so large robot packages do not exhaust connections.
        for(let start=0;start<wanted.length;start+=6)await Promise.all(wanted.slice(start,start+6).map(async path=>{
          const r=await fetch(new URL('urdf/'+path,location.href));if(!r.ok)throw Error('Missing resource: '+path);
          files.set(path,new File([await r.blob()],path.split('/').pop()));
        }));
      }else{
        const doc=new DOMParser().parseFromString(text,'application/xml');
        for(const element of doc.querySelectorAll('mesh[filename],texture[filename]')){
          const ref=element.getAttribute('filename');
          if(ref.startsWith('package://'))throw Error(this.message('loadCompleteFolder'));
          const url=new URL(ref,context?.base||location.href),r=await fetch(url);if(!r.ok)throw Error('Missing resource: '+ref);
          const path=normalize(entry.slice(0,entry.lastIndexOf('/')+1)+ref);files.set(path,new File([await r.blob()],path.split('/').pop()));
        }
      }
      files.set(entry,new File([text],entry.split('/').pop(),{type:'application/xml'}));
      return {entry,files};
    }
    async convert(text,context,target){
      const {entry,files}=await this.bundle(text,context),response=await this.request('convert',entry,files,target,context?.originalFormat?{format:context.originalFormat,warnings:context.formatReport?.warnings||[]}:null);
      const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement('a');
      anchor.href=url;anchor.download=(context?.name||'model').replace(/\.[^.]+$/,'')+'-'+target+'.zip';document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
      const report=response.headers.get('X-Conversion-Report');return report?JSON.parse(decodeURIComponent(report)):null;
    }
  };
})();
