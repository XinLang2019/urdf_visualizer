/* Local source handles: keep the selected file identity and commit editor text in place. */
(() => {
  'use strict';
  const normalized=text=>text.replace(/\r\n?/g,'\n');
  const types=[{description:'URDF / XML',accept:{'application/xml':['.urdf','.xml']}}];
  window.URDFSourceFiles=class {
    constructor(message) {this.message=message;this.local=new WeakMap();this.urls=new Map();}
    fromFile(file,handle=null) {
      if(!this.local.has(file))this.local.set(file,{name:file.name,file,handle,savedText:null,written:false});
      return this.local.get(file);
    }
    fromURL(url,text) {
      const key=String(url);
      if(!this.urls.has(key))this.urls.set(key,{name:decodeURIComponent(new URL(key).pathname.split('/').pop()),handle:null,written:false});
      const source=this.urls.get(key);source.savedText=normalized(text);return source;
    }
    async read(source) {
      if(source.handle)source.file=await source.handle.getFile();
      const text=normalized(await source.file.text());source.savedText=text;return text;
    }
    async readHandle(handle,prefix='') {
      if(handle.kind==='file') {
        const file=await handle.getFile();
        Object.defineProperty(file,'_relativePath',{value:prefix+file.name});
        this.fromFile(file,handle);return [file];
      }
      const pending=[];
      for await(const entry of handle.values())pending.push(this.readHandle(entry,prefix+handle.name+'/'));
      return (await Promise.all(pending)).flat();
    }
    async pickFiles() {
      const handles=await window.showOpenFilePicker({multiple:true,types:[{description:'URDF / XML / meshes / textures',accept:{'application/octet-stream':['.urdf','.xml','.stl','.dae','.obj','.gltf','.glb','.bin','.png','.jpg','.jpeg','.webp']}}],excludeAcceptAllOption:false});
      return (await Promise.all(handles.map(handle=>this.readHandle(handle)))).flat();
    }
    async pickDirectory() {return this.readHandle(await window.showDirectoryPicker({id:'urdf-models'}));}
    async save(source,text,name) {
      let handle=source.handle;
      if(!handle) {
        if(source.savedText!==null) {
          if(!window.showOpenFilePicker)throw new Error(this.message('saveUnsupported'));
          // A File input / HTTP response has no writable identity. Bind the original,
          // never create an unnoticed copy under its name in the Downloads folder.
          [handle]=await window.showOpenFilePicker({multiple:false,types:[{...types[0],description:`${this.message('chooseSource')}: ${source.name}`}],excludeAcceptAllOption:false});
          if(handle.name!==source.name)throw new Error(`${this.message('wrongSource')}: ${source.name}`);
        }else{
          if(!window.showSaveFilePicker)throw new Error(this.message('saveUnsupported'));
          handle=await window.showSaveFilePicker({suggestedName:name,types});
        }
      }
      if(handle.queryPermission&&await handle.queryPermission({mode:'readwrite'})!=='granted') {
        if(await handle.requestPermission({mode:'readwrite'})!=='granted')throw new Error(this.message('writeDenied'));
      }
      const disk=normalized(await (await handle.getFile()).text());
      if(source.savedText!==null&&disk!==source.savedText&&disk!==text)throw new Error(this.message('sourceChanged'));
      const stream=await handle.createWritable();
      try {await stream.write(text);await stream.close();}
      catch(error){try{await stream.abort();}catch{}throw error;}
      // Update the saved baseline only after the file stream has committed.
      source.handle=handle;source.name=handle.name;source.savedText=text;source.written=true;
      source.file=new File([text],handle.name,{type:'application/xml'});
      this.local.set(source.file,source);
      return source;
    }
  };
})();
