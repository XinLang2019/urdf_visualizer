"""STEP export with a constrained JS heap, real downloads and responsive controls."""
import json
import os
from pathlib import Path
import time
import zipfile

import browser_smoke as smoke
from model_formats_browser import import_files


def wait_download(output,name,timeout=180):
    deadline=time.time()+timeout;path=output/name
    while time.time()<deadline:
        if path.exists() and not Path(str(path)+'.crdownload').exists() and zipfile.is_zipfile(path):return path
        time.sleep(.1)
    raise AssertionError('Download did not finish: '+name)


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Target.setAutoAttach',{'autoAttach':True,'waitForDebuggerOnStart':False,'flatten':True})
    client.call('Page.navigate',{'url':os.environ.get('URDF_STEP_TEST_URL',url)})
    client.until('!!window.urdfWorkbench')
    client.call('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':str(output)})
    client.js('''window.stepDownloads=[];window.originalAnchorClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download){this.download=stepDownloads.length+'-'+this.download;stepDownloads.push({name:this.download})}return originalAnchorClick.call(this)};
window.stepReports=[];window.originalFormatRequest=URDFModelFormats.prototype.request;URDFModelFormats.prototype.request=async function(...args){const response=await originalFormatRequest.apply(this,args);if(args[3]==='step')stepReports.push(JSON.parse(decodeURIComponent(response.headers.get('X-Conversion-Report'))));return response};
window.stepFrames=0;window.stepTicks=0;window.stepMaxGap=0;window.stepLastTick=performance.now();setInterval(()=>{const now=performance.now();stepMaxGap=Math.max(stepMaxGap,now-stepLastTick);stepLastTick=now;stepTicks++},25);requestAnimationFrame(function tick(){stepFrames++;requestAnimationFrame(tick)});''')
    # Reproduce a local-folder import, using the full Go2 visual mesh package.
    source=smoke.ROOT/'urdf/go2';files={'go2/urdf/go2.urdf':(source/'urdf/go2.urdf').read_bytes()}
    for path in (source/'dae').glob('*.dae'):files['go2/dae/'+path.name]=path.read_bytes()
    import_files(client,files,'go2_description')
    client.js('window.stepOriginalRobot=urdfWorkbench.viewer.robot;window.stepOriginalSource=document.getElementById("source").value;document.getElementById("conversion-target").value="step"')
    # Cancel after the background worker has opened the output; leave no partial ZIP.
    client.js('document.getElementById("convert-model").click()')
    client.until('document.getElementById("conversion-status").textContent.includes("后台生成")',120)
    client.js('document.getElementById("cancel-conversion").click()')
    client.until('!document.getElementById("convert-model").disabled',3)
    smoke.check(client,'stepDownloads.length===0 && document.getElementById("conversion-status").textContent.includes("已取消")','cancelling stops the worker and does not download an incomplete file')
    client.until('''(async()=>{const d=await navigator.storage.getDirectory();for await(const name of d.keys())if(name.startsWith('urdf-step-'))return false;return true})()''',5)
    print('PASS cancellation removes temporary output and restores controls',flush=True)
    # Retry to completion. Sample both JS heaps while the real ZIP goes to disk;
    # do not turn a 180 MB download into base64 in the page or the DevTools socket.
    client.js('window.stepFrames=0;window.stepTicks=0;window.stepMaxGap=0;stepLastTick=performance.now();document.getElementById("convert-model").click()')
    client.until('document.getElementById("conversion-status").textContent.includes("后台生成")',120)
    client.js('window.stepFrames=0;window.stepTicks=0;window.stepMaxGap=0;stepLastTick=performance.now()')
    attached=[e['params'] for e in client.events if e.get('method')=='Target.attachedToTarget' and e['params']['targetInfo']['type']=='worker']
    assert attached,'No worker debugger session'
    session=attached[-1]['sessionId']
    peak_main=peak_worker=0;progress=[];deadline=time.time()+180
    while time.time()<deadline:
        info=client.js('({busy:document.getElementById("convert-model").disabled,progress:document.getElementById("conversion-progress").value,error:document.getElementById("conversion-status").classList.contains("error"),status:document.getElementById("conversion-status").textContent})')
        assert not info['error'],info
        peak_main=max(peak_main,client.call('Runtime.getHeapUsage')['usedSize'])
        if not info['busy']:break
        try:peak_worker=max(peak_worker,client.call('Runtime.getHeapUsage',session_id=session)['usedSize'])
        except RuntimeError:pass  # The worker terminates immediately after completion.
        progress.append(info['progress']);time.sleep(.1)
    else:raise AssertionError('STEP conversion timed out')
    assert progress==sorted(progress) and progress[-1]>.4,progress
    report=client.js('stepReports[0]');assert report['step']['worker'] and report['step']['storage']=='disk',report
    name=client.js('stepDownloads[0].name');path=wait_download(output,name)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())=={'model.step','conversion-report.json','README.txt'}
        with archive.open('model.step') as step:assert step.read(32).startswith(b'ISO-10303-21;')
        assert json.loads(archive.read('conversion-report.json'))==report
    activity=client.js('({frames:stepFrames,ticks:stepTicks,gap:stepMaxGap})')
    assert activity['frames']>10 and activity['ticks']>10 and activity['gap']<1500,activity
    assert peak_main<220*1024*1024 and peak_worker<220*1024*1024,(peak_main,peak_worker)
    print('PASS real local Go2 download with bounded heaps: '+json.dumps({'bytes':path.stat().st_size,'mainMiB':round(peak_main/1024**2,1),'workerMiB':round(peak_worker/1024**2,1),'activity':activity}),flush=True)
    smoke.check(client,'urdfWorkbench.viewer.robot===stepOriginalRobot && document.getElementById("source").value===stepOriginalSource','conversion preserves the loaded robot and editable source')
    # This mesh would allocate hundreds of thousands of STEP faces; refuse before
    # expanding them, keep the model usable, and name the simplification action.
    urdf='<robot name="too_dense"><link name="base"><visual><geometry><mesh filename="dense.stl"/></geometry></visual></link></robot>'
    # Inject the input package at the file boundary; drawing 250,001 overlapping
    # triangles would test the software GPU rather than STEP's preflight guard.
    client.js('''(()=>{const bytes=new Uint8Array(84+250001*50),view=new DataView(bytes.buffer);view.setUint32(80,250001,true);const files=new Map([['dense.urdf',new File([SOURCE_XML],'dense.urdf')],['dense.stl',new File([bytes],'dense.stl')]]);window.originalFormatBundle=URDFModelFormats.prototype.bundle;URDFModelFormats.prototype.bundle=async()=>({entry:'dense.urdf',files});})()'''.replace('SOURCE_XML',json.dumps(urdf)))
    client.js('document.getElementById("convert-model").click()')
    client.until('document.getElementById("conversion-status").classList.contains("error") && !document.getElementById("convert-model").disabled',60)
    smoke.check(client,'document.getElementById("conversion-status").textContent.includes("Mesh Simplifier") && stepDownloads.length===1','oversized individual meshes fail clearly without starting a huge STEP or crashing')
    client.js('URDFModelFormats.prototype.bundle=originalFormatBundle')
    # Block only the worker's temporary storage, as in a restrictive browser profile.
    client.js('''window.originalStepWorker=Worker;window.blockedWorkerURLs=[];window.Worker=class extends originalStepWorker{constructor(url){const script="Object.defineProperty(navigator.storage,'getDirectory',{value:async()=>{throw Error('Storage blocked')}});importScripts("+JSON.stringify(String(url))+");";const blob=URL.createObjectURL(new Blob([script],{type:'text/javascript'}));blockedWorkerURLs.push(blob);super(blob)}};document.getElementById('convert-model').click()''')
    client.until('!document.getElementById("convert-model").disabled && document.getElementById("conversion-status").classList.contains("error")',90)
    smoke.check(client,'document.getElementById("conversion-status").textContent.includes("临时存储不可用") && stepDownloads.length===1','large exports stop safely when temporary storage is unavailable')
    client.js('''(()=>{const xml='<robot name="fallback"><link name="base"><visual><geometry><box size=".1 .2 .3"/></geometry></visual></link></robot>';URDFModelFormats.prototype.bundle=async()=>({entry:'fallback.urdf',files:new Map([['fallback.urdf',new File([xml],'fallback.urdf')]])});document.getElementById('convert-model').click()})()''')
    client.until('stepDownloads.length===2 && !document.getElementById("convert-model").disabled',30)
    assert client.js('stepReports[1].step.storage')=='memory'
    fallback=wait_download(output,client.js('stepDownloads[1].name'))
    with zipfile.ZipFile(fallback) as archive:assert archive.testzip() is None
    client.js('Worker=originalStepWorker;URDFModelFormats.prototype.bundle=originalFormatBundle;blockedWorkerURLs.forEach(u=>URL.revokeObjectURL(u))')
    print('PASS unavailable-storage guard and bounded small-file memory fallback',flush=True)
    requests=[e['params']['request']['url'] for e in client.events if e.get('method')=='Network.requestWillBeSent']
    assert not any('/api/formats/' in request for request in requests)
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    print('PASS bounded memory, UI responsiveness, cancellable retry, clean ZIP CRCs and oversized mesh protection',flush=True)
    client.screenshot(str(output/'step-stability.png'))


if __name__=='__main__':
    os.environ['URDF_TEST_FORMATS']='0';os.environ.setdefault('URDF_TEST_HEAP_MB','256')
    smoke.run=run;smoke.main()
