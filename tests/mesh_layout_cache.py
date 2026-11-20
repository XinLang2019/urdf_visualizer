"""Warm-cache deployment and exact mesh selection regression in real Edge/Chrome."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlsplit
import browser_smoke as smoke
from model_interaction import rendered_frame


class CachedAssets(SimpleHTTPRequestHandler):
    legacy={
        '/workbench.css':b'/* cached old CSS */ .mesh-tools { margin:0 } #mesh-tools-title { font-size:30px }',
        '/workbench.js':b'window.__legacyMeshScript=true;',
    }
    hits={}
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(smoke.ROOT),**kwargs)
    def do_GET(self):
        url=urlsplit(self.path)
        if url.path=='/cache-start':
            data=b'<!doctype html><title>Cache fixture</title>'
            self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        if not url.query and url.path in self.legacy:
            self.hits[url.path]=self.hits.get(url.path,0)+1;data=self.legacy[url.path]
            self.send_response(200);self.send_header('Content-Type','text/css' if url.path.endswith('.css') else 'text/javascript');self.send_header('Cache-Control','max-age=3600');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        return super().do_GET()
    def log_message(self,*args):
        pass


def stl(faces):
    return 'solid test\n'+'\n'.join('facet normal 0 0 0\nouter loop\n'+'\n'.join('vertex '+' '.join(map(str,p)) for p in f)+'\nendloop\nendfacet' for f in faces)+'\nendsolid test'


def import_meshes(client,files):
    client.js('''(()=>{const d=new DataTransfer();for(const[path,text]of FILES){const f=new File([text],path.split('/').pop());Object.defineProperty(f,'webkitRelativePath',{value:path});d.items.add(f);}const i=document.getElementById('folder-input');i.files=d.files;i.dispatchEvent(new Event('change'));})()'''.replace('FILES',json.dumps(list(files.items()))))


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':942,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url+'cache-start'});client.until('document.title==="Cache fixture"')
    for _ in range(2):
        client.js('Promise.all(["workbench.css","workbench.js"].map(p=>fetch(p,{cache:"force-cache"}).then(r=>r.text())))')
    assert CachedAssets.hits=={'/workbench.css':1,'/workbench.js':1},CachedAssets.hits
    print('PASS old unversioned CSS and workbench JS are genuinely retained in browser cache',flush=True)
    client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    smoke.check(client,'!window.__legacyMeshScript','updated page loads the current workbench even with a warm legacy cache')
    smoke.check(client,'document.querySelector("#mesh-source option").textContent==="未加载 STL / OBJ / DAE"&&document.getElementById("mesh-source").disabled','empty mesh selector displays a clear initialized state')
    for width in [300,440]:
        client.js('document.body.style.setProperty("--side",'+json.dumps(str(width)+'px')+')')
        rendered_frame(client)
        smoke.check(client,'(()=>{const a=document.querySelector(".format-tools:not(.mesh-tools)"),b=document.querySelector(".mesh-tools"),ar=a.getBoundingClientRect(),br=b.getBoundingClientRect(),s=document.getElementById("mesh-source").getBoundingClientRect(),button=document.getElementById("mesh-open").getBoundingClientRect();return Math.abs(ar.left-br.left)<1&&Math.abs(ar.width-br.width)<1&&getComputedStyle(b.querySelector("h3")).fontSize===getComputedStyle(a.querySelector("h3")).fontSize&&s.width>100&&s.right<button.left&&button.height<=36&&br.right<=document.getElementById("files-panel").getBoundingClientRect().right})()','mesh controls align with format controls at '+str(width)+'px sidebar width')
    a,b,c,d=[0,0,0],[1,0,0],[0,1,0],[0,0,1]
    first=stl([[a,c,b],[a,b,d],[a,d,c],[b,c,d]]);second=stl([[a,b,c],[b,[1,1,0],c]])
    files={'robot/meshes/left/shared.stl':first,'robot/meshes/right/shared.stl':second,'robot/notes.xml':'<metadata/>'}
    import_meshes(client,files);client.until('urdfWorkbench.meshSimplifier.files.length===2')
    smoke.check(client,'document.getElementById("mesh-source").options.length===3&&document.getElementById("mesh-open").disabled','multiple meshes require an explicit choice and omit non-mesh metadata')
    for path,faces in [('robot/meshes/left/shared.stl',4),('robot/meshes/right/shared.stl',2)]:
        client.js('document.getElementById("mesh-source").value='+json.dumps(path)+';document.getElementById("mesh-source").dispatchEvent(new Event("change"));document.getElementById("mesh-open").click()')
        client.until('!!urdfWorkbench.meshSimplifier.original&&!urdfWorkbench.meshSimplifier.busy')
        smoke.check(client,'urdfWorkbench.meshSimplifier.original.faces==='+str(faces)+'&&document.getElementById("mesh-name").textContent==='+json.dumps(path),'duplicate basenames load the exact selected relative path: '+path)
        client.js('document.getElementById("mesh-close").click()')
    import_meshes(client,{'another/meshes/fresh.stl':second});client.until('urdfWorkbench.meshSimplifier.files.length===1')
    smoke.check(client,'document.getElementById("mesh-source").value==="another/meshes/fresh.stl"&&document.getElementById("mesh-source").options.length===2&&!document.getElementById("mesh-open").disabled','switching folders removes stale file choices and selects the single new mesh')
    client.screenshot(str(output/'mesh-layout-cache-fixed.png'))
    assert CachedAssets.hits=={'/workbench.css':1,'/workbench.js':1},CachedAssets.hits
    exceptions=[e for e in client.events if e.get('method')=='Runtime.exceptionThrown'];assert not exceptions,exceptions
    print('PASS current CSS, scripts and worker versions bypass old assets without clearing browser cache',flush=True)


if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',0),CachedAssets)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    smoke.run=lambda c,url,out:run(c,'http://127.0.0.1:'+str(server.server_port)+'/',out)
    try:smoke.main()
    finally:server.shutdown();server.server_close();thread.join()
