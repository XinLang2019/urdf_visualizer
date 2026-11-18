"""Actual multipart APIs + Edge/Chrome: native models, six conversions and reimport."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
import browser_smoke as smoke
from test_model_formats import URDF, physical
from model_interaction import rendered_frame

BROWSER_FORMATS=os.environ.get('URDF_TEST_BROWSER_FORMATS')=='1'

def verify_readable_export(package):
    """Downloaded source is readable, including large inline mesh arrays."""
    import xml.etree.ElementTree as ET
    for name,data in package.items():
        if name not in ['model.urdf','model.xml','model.usda']:continue
        text=data.decode()
        assert text.endswith('\n') and '\t' not in text
        if name.endswith('.usda'):
            assert '\n    metersPerUnit = 1\n' in text
            assert '\n    def Xform "Links" {\n        def Xform ' in text
            assert '\n                point3f[] points = [\n                    (' in text
            # Numeric mesh rows are wrapped; a dense mesh must not become one huge line.
            rows=[line for line in text.splitlines() if line.startswith(' '*20)]
            assert rows and max(map(len,rows))<=120
        else:
            root=ET.fromstring(text)
            def nesting(node,depth=0):
                if len(node):
                    assert node.text=='\n'+'  '*(depth+1),(name,node.tag,node.text)
                    for child in node:nesting(child,depth+1)
                    assert node[-1].tail=='\n'+'  '*depth
            nesting(root)


def verify_browser_package(package):
    """Independent native SDKs check browser-generated files and physical frames."""
    import model_formats as formats
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='browser-format-physics-') as folder:
        root=Path(folder);(root/'original.urdf').write_text(URDF)
        original=formats.read_model(root,root/'original.urdf')[0]
        for name,value in package.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        entry=next(root/name for name in package if name in ['model.urdf','model.xml','model.usda'])
        actual=formats.read_model(root,entry)[0]
        if entry.suffix=='.xml':
            import mujoco
            compiled=mujoco.MjModel.from_xml_path(str(entry))
            assert abs(compiled.body_mass.sum()-3)<1e-6
        for angle in [-.7,0,.8]:
            a,b=physical(original,{'hinge':angle}),physical(actual,{'hinge':angle})
            assert set(a)==set(b)
            for name in a:
                for k in range(4):np.testing.assert_allclose(a[name][k],b[name][k],atol=5e-6)


def import_files(client,files,expected):
    # Formatted Go2 USDA exceeds DevTools' 100 MiB message limit when base64 encoded.
    # Transfer chunks, then upload the same complete File through the real folder input.
    client.js('window.formatUploadFiles=[]')
    for path,value in files.items():
        data=value if isinstance(value,bytes) else value.encode()
        client.js('window.formatUploadChunks=[]')
        for offset in range(0,len(data),1024*1024):
            chunk=base64.b64encode(data[offset:offset+1024*1024]).decode()
            client.js('formatUploadChunks.push(Uint8Array.from(atob('+json.dumps(chunk)+'),c=>c.charCodeAt(0)))')
        client.js('''(()=>{const path=PATH,f=new File(formatUploadChunks,path.split('/').pop());Object.defineProperty(f,'webkitRelativePath',{value:path});formatUploadFiles.push(f);delete window.formatUploadChunks})()'''.replace('PATH',json.dumps(path)))
    client.js('''(()=>{window.beforeFormatImport=urdfWorkbench.viewer.robot;const d=new DataTransfer();for(const f of formatUploadFiles)d.items.add(f);delete window.formatUploadFiles;const input=document.getElementById('folder-input');input.files=d.files;input.dispatchEvent(new Event('change'));})()''')
    client.until('urdfWorkbench.viewer.robot!==beforeFormatImport && urdfWorkbench.viewer.robot?.robotName==='+json.dumps(expected)+' && !urdfWorkbench.loading',90)
    smoke.check(client,'!document.querySelector("#status.error")','native import loads geometry without failed asset requests: '+expected)
    rendered_frame(client)


def convert(client,target):
    count=client.js('formatDownloads.length')
    client.js('document.getElementById("conversion-target").value='+json.dumps(target)+';document.getElementById("convert-model").click()')
    client.until('formatDownloads.length>'+str(count)+' || document.getElementById("conversion-status").classList.contains("error")',120)
    error=client.js('document.getElementById("conversion-status").classList.contains("error")?document.getElementById("conversion-status").textContent:null')
    assert not error,error
    item=client.js('formatDownloads['+str(count)+']')
    raw=base64.b64decode(item['data'])
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        report=json.loads(archive.read('conversion-report.json'))
        assert report['targetFormat']==target
        result={name:archive.read(name) for name in archive.namelist() if name not in ['README.txt','conversion-report.json']}
        assert report['entry'] in result
        if target=='usd':assert result[report['entry']].startswith(b'#usda') if BROWSER_FORMATS else result[report['entry']][:8]==b'PXR-USDC'
        if BROWSER_FORMATS:
            assert report['engine']=='browser'
            verify_readable_export(result)
    client.until('!document.getElementById("convert-model").disabled')
    return result,report


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False})
    client.call('Page.navigate',{'url':url if BROWSER_FORMATS else url+'?formats=native'});client.until('!!window.urdfWorkbench')
    client.js('''window.formatDownloads=[];window.formatClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.blob()).then(blob=>new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(blob);})).then(data=>formatDownloads.push({name,data}));}else formatClick.call(this)}''')
    import_files(client,{'original.urdf':URDF},'format_fixture')
    smoke.check(client,'document.getElementById("filename").value==="original.urdf"','ordinary URDF source remains the editable original')
    native={}
    for target in ['mjcf','usd']:
        package,report=convert(client,target);native[target]=package
        smoke.check(client,'!document.getElementById("conversion-status").classList.contains("error")','URDF → '+target.upper()+' produces a resource package')
        assert report['links']==2 and report['joints']==1
        if BROWSER_FORMATS:verify_browser_package(package)
    for source,package in native.items():
        import_files(client,package,'format_fixture')
        smoke.check(client,'document.getElementById("filename").value.endsWith(".viewer.urdf") && document.getElementById("save-state").textContent.includes("未保存")','native '+source.upper()+' exposes an explicit URDF working copy instead of overwriting the source')
        smoke.check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.hinge.limit.lower+1)<1e-5 && Math.abs(urdfWorkbench.viewer.robot.joints.hinge.limit.upper-1.2)<1e-5','native '+source.upper()+' retains signed radian limits')
        client.js('(()=>{document.getElementById("tab-joints").click();const slider=document.querySelector("#joint-list li[data-name=hinge] input[type=range]");slider.value=.4;slider.dispatchEvent(new Event("input"));})()')
        smoke.check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.hinge.angle-.4)<1e-7','native '+source.upper()+' supports existing joint controls')
        client.js('document.getElementById("tab-files").click()')
        for target in ['urdf','usd' if source=='mjcf' else 'mjcf']:
            converted,report=convert(client,target)
            if BROWSER_FORMATS:verify_browser_package(converted)
            smoke.check(client,'!document.getElementById("conversion-status").classList.contains("error")',source.upper()+' → '+target.upper()+' produces a valid native file and conversion report')
            # Ensure every resource referenced by exported URDF is included.
            if target=='urdf':
                import xml.etree.ElementTree as ET
                doc=ET.fromstring(converted['model.urdf'])
                for mesh in doc.findall('.//mesh'):assert mesh.get('filename') in converted
        client.screenshot(str(output/('formats-'+source+'.png')))
    # Reimport an exported binary USD package and save a working copy to a new file.
    import_files(client,native['usd'],'format_fixture')
    count=client.js('formatDownloads.length')
    client.js("window.beforeCopySource=document.getElementById('source').value;document.getElementById('save').click()")
    client.until('formatDownloads.length>'+str(count)+' && document.getElementById("save-state").textContent==="副本已导出"',90)
    item=client.js('formatDownloads['+str(count)+']')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(item['data']))) as archive:
        assert archive.read('model.urdf').decode().startswith('<?xml')
        import xml.etree.ElementTree as ET
        for mesh in ET.fromstring(archive.read('model.urdf')).findall('.//mesh'):assert mesh.get('filename') in archive.namelist()
    smoke.check(client,'document.getElementById("source").value===beforeCopySource && document.getElementById("save").textContent==="导出 URDF 包"','saving a USD working copy exports XML with all mesh resources and leaves the native source untouched')
    # Missing / unsupported native resources retain the currently loaded model.
    bad='<mujoco><asset><mesh name="missing" file="missing.stl"/></asset><worldbody><body><geom type="mesh" mesh="missing"/></body></worldbody></mujoco>'
    client.js('''(()=>{window.retainedFormatRobot=urdfWorkbench.viewer.robot;const d=new DataTransfer();d.items.add(new File([BAD],'broken.xml'));const i=document.getElementById('file-input');i.files=d.files;i.dispatchEvent(new Event('change'))})()'''.replace('BAD',json.dumps(bad)))
    client.until('!!document.querySelector("#status.error")',30)
    smoke.check(client,'urdfWorkbench.viewer.robot===retainedFormatRobot && document.getElementById("editor-error").textContent.includes("Missing")','missing native mesh resources retain the previously loaded model')
    # Both native examples are loadable without a local upload.
    for example,expected in [('examples/formats/two-link.xml','two_link_mjcf'),('examples/formats/two-link.usda','two_link_usd')]:
        client.js('document.getElementById("example-select").value='+json.dumps(example)+';document.getElementById("load-example").click()')
        client.until('urdfWorkbench.viewer.robot?.robotName==='+json.dumps(expected)+' && !urdfWorkbench.loading',90)
        smoke.check(client,'!document.querySelector("#status.error")','native example loads: '+example)
    if BROWSER_FORMATS:
        from browser_format_cases import check_browser_cases
        check_browser_cases(client)
    # Real bundled Go2: DAE coordinates, native centering and USD geometry.
    client.js("""window.formatBounds=()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),out={};urdfWorkbench.viewer.world.updateWorldMatrix(true,true);for(const [name,link]of Object.entries(urdfWorkbench.viewer.robot.links)){const box=new T.Box3();for(const child of link.children)if(child.isURDFVisual)box.expandByObject(child,true);if(!box.isEmpty())out[name]=[...box.min.toArray(),...box.max.toArray()];}return out}""")
    for target in ['mjcf','usd']:
        client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
        client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description" && !urdfWorkbench.loading',60)
        client.js('window.originalGo2FormatBounds=formatBounds()')
        go2_package,go2_report=convert(client,target)
        import_files(client,go2_package,'go2_description')
        smoke.check(client,"""(()=>{const actual=formatBounds();return Object.entries(originalGo2FormatBounds).every(([name,b])=>actual[name]&&b.every((v,i)=>Math.abs(v-actual[name][i])<.0002))})()""",'actual Go2 → '+target.upper()+' preserves precise visible link bounds and native mesh transforms')
        smoke.check(client,'document.querySelectorAll("#joint-list li[data-type=revolute]").length===12','actual Go2 '+target.upper()+' retains all 12 movable joints')
    client.js('document.getElementById("conversion-target").value="usd"')
    for width,height in [(390,844),(1897,942)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        smoke.check(client,'document.documentElement.scrollWidth<=innerWidth && document.getElementById("files-panel").scrollWidth<=document.getElementById("files-panel").clientWidth','format tools fit the '+str(width)+'px viewport')
    client.screenshot(str(output/'format-conversion.png'))
    if BROWSER_FORMATS:
        requests=[e['params']['request']['url'] for e in client.events if e.get('method')=='Network.requestWillBeSent']
        assert not any('/api/formats/' in u for u in requests),requests
        print('PASS static hosting uses no Python API, uploads or external conversion services',flush=True)
    client.js('HTMLAnchorElement.prototype.click=formatClick')
    exceptions=[e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    assert not exceptions,exceptions
    print('PASS '+('static browser import, all six conversions, text USDA' if BROWSER_FORMATS else 'native multipart import, all six conversions, binary USD')+', packaged meshes, joint controls, safe working-copy saves and errors',flush=True)

if __name__=='__main__':
    os.environ['URDF_TEST_FORMATS']='0' if BROWSER_FORMATS else '1';smoke.run=run;smoke.main()
