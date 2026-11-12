"""Real WebGL integration checks. Requires Python 3 and Chrome / Chromium."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from cdp import CDP
from model_interaction import check_model_interaction
from helper_visuals import check_helper_visuals
from link_measurement import check_link_measurement
from surface_measurement import check_surface_measurement
from link_navigation import check_link_navigation
from joint_angle_editor import check_joint_angle_editor
from workspace_layout import check_workspace_layout
from statics_analysis import check_statics_analysis
from slope_statics import check_slope_statics, check_terrain_lifecycle
from support_fitting import check_support_fitting, check_go2_support_fit
from statics_labels import check_statics_labels
from editor_save import check_editor_save
from wheel_contacts import check_wheel_contacts
from trajectory_editor import check_trajectory_editor

ROOT = Path(__file__).resolve().parents[1]
BROWSER = os.environ.get('CHROME_BIN') or next((shutil.which(name) for name in ['google-chrome', 'chromium', 'chromium-browser'] if shutil.which(name)), None)
if not BROWSER:
    sys.exit('Chrome / Chromium not found. Set CHROME_BIN to its executable.')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def check(client, expression, message):
    assert client.js(expression), message
    print('PASS', message, flush=True)


def click(client, element_id):
    client.js(f'document.getElementById({json.dumps(element_id)}).click()')


def run(client, url, output):
    client.call('Runtime.enable')
    client.call('Page.enable')
    client.call('Emulation.setDeviceMetricsOverride', {'width': 1897, 'height': 942, 'deviceScaleFactor': 1, 'mobile': False})
    client.call('Page.navigate', {'url': url})
    client.until('!!window.urdfWorkbench')
    check(client, '!document.querySelector("#status.error")', 'WebGL initializes without external dependencies')
    check(client, 'document.getElementById("graph-measure").disabled', 'link measurement is disabled until a model is loaded')
    client.screenshot(str(output / 'empty.png'))
    for model in ['anymal_c/urdf/anymal_c', 'anymal_b/urdf/anymal_b', 'g1/urdf/g1', 'cassie/urdf/cassie', 'go2/urdf/go2']:
        client.js('document.getElementById("example-select").value=' + json.dumps('urdf/' + model + '.urdf'))
        click(client, 'load-example')
        client.until('!urdfWorkbench.loading', 55)
        check(client, '!!urdfWorkbench.viewer.robot && !document.querySelector("#status.error")', 'bundled model loads: ' + model)
    check(client, 'Object.keys(urdfWorkbench.viewer.robot.joints).length===28', 'Go2 exposes all joints')
    check(client, 'JSON.stringify([...document.querySelectorAll("#joint-list li")].filter(row=>!row.hidden).map(row=>row.dataset.name))===JSON.stringify(["FL_hip_joint","FL_thigh_joint","FL_calf_joint","FR_hip_joint","FR_thigh_joint","FR_calf_joint","RL_hip_joint","RL_thigh_joint","RL_calf_joint","RR_hip_joint","RR_thigh_joint","RR_calf_joint"])', 'joint cards follow complete leg chains in model order')
    check(client, '!!document.querySelector("#joint-list .joint-effort") && !!document.querySelector("#joint-list .joint-velocity") && !!document.querySelector("#joint-list .joint-lower") && !!document.querySelector("#joint-list .joint-upper")', 'joint cards expose effort, velocity and both limits')
    check(client, 'document.querySelector("#file-tree .file-entry.selected").dataset.path==="go2/urdf/go2.urdf" && document.querySelectorAll("#file-tree .file-folder").length>=3 && getComputedStyle(document.querySelector(".import-area")).display==="none"', 'loaded model shows folders and highlights the active URDF')
    check(client, 'document.getElementById("source-highlight").textContent.trim()===document.getElementById("source").value.trim() && !!document.querySelector("#source-highlight .xml-tag") && !!document.querySelector("#source-highlight .xml-comment")', 'safe XML highlighting preserves editor contents')
    check_workspace_layout(client, output, check)
    collapsed_nodes = client.js('document.querySelectorAll(".structure-node").length')
    click(client, 'graph-fixed')
    check(client, 'document.querySelectorAll(".structure-node").length===Object.keys(urdfWorkbench.viewer.robot.links).length+Object.keys(urdfWorkbench.viewer.robot.joints).length+1', 'full topology includes every link and joint')
    click(client, 'graph-fixed')
    check(client, f'document.querySelectorAll(".structure-node").length==={collapsed_nodes}', 'fixed attachments can be collapsed')
    # Real pointer events test SVG selection and capture, not just DOM handlers.
    node = client.js('(()=>{const r=document.querySelector(".structure-node[data-model-root=true] rect").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    for event_type in ['mousePressed', 'mouseReleased']:
        client.call('Input.dispatchMouseEvent', {'type': event_type, 'x': node['x'], 'y': node['y'], 'button': 'left', 'clickCount': 1})
    check(client, 'document.querySelector(".structure-node[data-model-root=true]").classList.contains("selected")', 'topology nodes select through real pointer input')
    before = client.js('document.querySelector("#tree>svg>g").getAttribute("transform")')
    host = client.js('(()=>{const r=document.getElementById("tree").getBoundingClientRect();return {x:r.x+10,y:r.y+10}})()')
    client.call('Input.dispatchMouseEvent', {'type': 'mousePressed', 'x': host['x'], 'y': host['y'], 'button': 'left', 'buttons': 1, 'clickCount': 1})
    client.call('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'x': host['x']+35, 'y': host['y']+20, 'button': 'left', 'buttons': 1})
    client.call('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'x': host['x']+35, 'y': host['y']+20, 'button': 'left'})
    check(client, 'document.querySelector("#tree>svg>g").getAttribute("transform")!=='+json.dumps(before), 'topology drag pans the diagram')
    click(client, 'graph-fit')
    before = client.js('document.querySelector("#tree>svg>g").getAttribute("transform")')
    client.call('Input.dispatchMouseEvent', {'type': 'mouseWheel', 'x': host['x']+100, 'y': host['y']+80, 'deltaX': 0, 'deltaY': -180})
    client.until('document.querySelector("#tree>svg>g").getAttribute("transform")!=='+json.dumps(before))
    print('PASS topology wheel zooms the diagram', flush=True)
    click(client, 'graph-fit')
    # Raycast two real pointer events against the model surface.
    point = client.js("""(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer,b=new T.Box3();v.robot.traverse(x=>{if(x.isURDFVisual)b.expandByObject(x)});const p=b.getCenter(new T.Vector3()).project(v.camera);return {x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2}})()""")
    for dx in [-8, 8]:
        for event_type in ['mousePressed', 'mouseReleased']:
            client.call('Input.dispatchMouseEvent', {'type': event_type, 'x': point['x'] + dx, 'y': point['y'], 'button': 'left', 'clickCount': 1, 'modifiers': 9})
    check(client, '!document.getElementById("measure-result").hidden && parseFloat(document.querySelector("#measure-result span").textContent)>0', 'Shift + click measures real model surfaces')
    client.js('document.querySelector("#measure-result button").click()')
    check(client, 'document.getElementById("measure-result").hidden && !document.getElementById("status-text").textContent.includes("第二个点")', 'clearing measurement resets the status')
    client.screenshot(str(output / 'go2.png'))
    client.js('''window.testRow=document.querySelector('#joint-list li[data-name="FL_hip_joint"]');
      testRow.querySelector('input[type=range]').value=.3;
      testRow.querySelector('input[type=range]').dispatchEvent(new Event('input'));''')
    click(client, 'unit-deg')
    check(client, 'Math.abs(Number(testRow.querySelector("input[type=number]").value)-17.2)<.01', 'radians convert to degrees')
    client.js('testRow.querySelector("input[type=number]").value=999;testRow.querySelector("input[type=number]").dispatchEvent(new Event("change"))')
    check(client, 'urdfWorkbench.viewer.robot.joints.FL_hip_joint.angle===urdfWorkbench.viewer.robot.joints.FL_hip_joint.limit.upper', 'joint input obeys limits')
    click(client, 'limits')
    client.js('testRow.querySelector("input[type=number]").value=120;testRow.querySelector("input[type=number]").dispatchEvent(new Event("change"))')
    check(client, 'urdfWorkbench.viewer.robot.joints.FL_hip_joint.angle>2', 'limits can be disabled')
    click(client, 'limits')
    click(client, 'reset-joints')
    check_trajectory_editor(client, output, check)
    for key in ['visual', 'collision', 'inertia', 'com', 'axes', 'jointAxes', 'shadow', 'lighting', 'grid']:
        client.js('document.querySelector(' + json.dumps(f'[data-display="{key}"]') + ').click()')
    check(client, '!urdfWorkbench.display.visual && urdfWorkbench.display.collision && !urdfWorkbench.display.grid', 'display controls change render state')
    for key in ['visual', 'collision', 'inertia', 'com', 'axes', 'jointAxes', 'shadow', 'lighting', 'grid']:
        client.js('document.querySelector(' + json.dumps(f'[data-display="{key}"]') + ').click()')
    client.js('window.oldRobot=urdfWorkbench.viewer.robot;window.originalText=document.getElementById("source").value')
    for bad_xml in ['<robot>', '<robot name="bad"><link name="a"/><link name="b"/></robot>']:
        client.js('document.getElementById("source").value=' + json.dumps(bad_xml) + ';document.getElementById("source").dispatchEvent(new Event("input"))')
        click(client, 'reload')
        check(client, 'oldRobot===urdfWorkbench.viewer.robot && !document.getElementById("editor-error").hidden', 'invalid XML or topology retains the model')
    client.js('document.getElementById("source").value=originalText;document.getElementById("source").dispatchEvent(new Event("input"))')
    click(client, 'reload')
    client.until('!urdfWorkbench.loading')
    check(client, 'oldRobot!==urdfWorkbench.viewer.robot', 'valid editor changes rebuild the model')
    client.js('document.querySelector("#tree .structure-node[data-model-root=true]").dispatchEvent(new MouseEvent("click",{bubbles:true}))')
    check(client, '!document.getElementById("selection-info").hidden', 'structure selection shows model details')
    client.js('document.querySelector("#tree .structure-node[data-model-root=true]").dispatchEvent(new MouseEvent("contextmenu",{bubbles:true,cancelable:true}))')
    check(client, '!urdfWorkbench.viewer.robot.visible', 'structure visibility hides the subtree')
    client.js('document.querySelector("#tree .structure-node[data-model-root=true]").dispatchEvent(new MouseEvent("contextmenu",{bubbles:true,cancelable:true}))')
    # Import browser File objects using directory-relative paths and package:// meshes.
    xml = '<robot name="local_test"><link name="base"><visual><geometry><mesh filename="package://demo/meshes/part.obj"/></geometry></visual></link><link name="arm"><visual><geometry><box size=".1 .1 .4"/></geometry></visual></link><joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>'
    obj = 'v -.2 -.2 0\nv .2 -.2 0\nv 0 .2 0\nf 1 2 3\n'
    files = [('demo/urdf/test.urdf', xml.replace('><', '>\r\n<')), ('demo/meshes/part.obj', obj)]
    script = '''(()=>{const transfer=new DataTransfer();for(const [path,text]of FILES){const f=new File([text],path.split('/').pop());Object.defineProperty(f,'webkitRelativePath',{value:path});transfer.items.add(f);}const input=document.getElementById('folder-input');input.files=transfer.files;input.dispatchEvent(new Event('change'));})()'''
    client.js(script.replace('FILES', json.dumps(files)))
    client.until('urdfWorkbench.viewer.robot?.robotName==="local_test" && !urdfWorkbench.loading')
    check(client, '!document.querySelector("#status.error")', 'local folder package:// mesh resolves')
    # No false dirty flag from CRLF XML after textarea newline normalization.
    check(client, 'document.getElementById("save-state").textContent==="已应用"', 'CRLF files remain clean after import')
    # Two URDFs present a model picker rather than choosing one silently.
    files.append(('demo/urdf/other.urdf', xml.replace('local_test', 'other_test')))
    client.js(script.replace('FILES', json.dumps(files)))
    client.until('!!document.querySelector("dialog[open]")')
    check(client, 'document.querySelector("dialog[open]").querySelectorAll("button").length===2', 'multiple URDFs open a model picker')
    client.js('[...document.querySelector("dialog[open]").querySelectorAll("button")].find(button=>button.textContent.includes("other.urdf")).click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="other_test" && !urdfWorkbench.loading')
    check(client, 'document.querySelector("#file-tree .file-entry.selected").dataset.path==="demo/urdf/other.urdf"', 'model picker highlights the chosen source file')
    client.js('document.querySelector(`#file-tree button[data-path="demo/urdf/test.urdf"]`).click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="local_test" && !urdfWorkbench.loading')
    check(client, '!document.querySelector("#status.error") && document.querySelector("#file-tree .file-entry.selected").dataset.path==="demo/urdf/test.urdf"', 'switching URDFs retains shared mesh resources')
    click(client, 'tab-files')
    click(client, 'file-tools-toggle')
    check(client, 'getComputedStyle(document.querySelector(".import-area")).display!=="none"', 'loaded file panel can reopen import controls')
    click(client, 'file-tools-toggle')
    check_model_interaction(client, output, check)
    check_helper_visuals(client, output, check)
    check_link_measurement(client, output, check)
    check_surface_measurement(client, output, check)
    check_link_navigation(client, output, check)
    check_joint_angle_editor(client, output, check)
    check_wheel_contacts(client, output, check)
    check_slope_statics(client, output, check)
    check_support_fitting(client, output, check)
    check_go2_support_fit(client, output, check)
    check_terrain_lifecycle(client, output, check)
    check_statics_analysis(client, output, check)
    check_statics_labels(client, output, check)
    check_editor_save(client, output, check)
    # A URDF with no visual geometry still leaves finite camera and ground values.
    client.js('document.getElementById("source").value=' + json.dumps('<robot name="empty_visual"><link name="base"/></robot>') + ';document.getElementById("source").dispatchEvent(new Event("input"))')
    click(client, 'reload')
    client.until('!urdfWorkbench.loading')
    check(client, 'urdfWorkbench.viewer.camera.position.toArray().every(Number.isFinite) && Number.isFinite(urdfWorkbench.viewer.plane.position.y)', 'empty geometry retains a valid camera')
    for width, height in [(1280, 800), (390, 844)]:
        client.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False})
        time.sleep(.2)
        check(client, 'document.body.scrollWidth===innerWidth', f'no viewport overflow at {width}px')
        if width == 390:
            client.js('document.querySelector("[data-panel=editor]").click()')
            check(client, 'getComputedStyle(document.getElementById("editor-panel")).display!=="none" && getComputedStyle(document.getElementById("files-panel")).display==="none"', 'mobile panel switching works')
        client.screenshot(str(output / f'layout-{width}.png'))
    click(client, 'language')
    check(client, 'document.documentElement.lang==="en"', 'English UI switches')
    click(client, 'theme')
    check(client, 'document.body.classList.contains("light")', 'light theme switches')
    exceptions = [e for e in client.events if e.get('method') == 'Runtime.exceptionThrown']
    assert not exceptions, exceptions
    print('PASS no uncaught browser exceptions', flush=True)


def main():
    server_port, debug_port = free_port(), free_port()
    url = f'http://127.0.0.1:{server_port}/'
    output = Path(tempfile.mkdtemp(prefix='urdf-viewer-checks-'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix='urdf-viewer-chrome-') as profile, open(output / 'browser.log', 'w') as log:
        server_command=[sys.executable,str(ROOT/'planner_server.py'),'--port',str(server_port)] if os.environ.get('URDF_TEST_PLANNER')=='1' else [sys.executable,'-m','http.server',str(server_port),'--bind','127.0.0.1']
        server = subprocess.Popen(server_command, cwd=ROOT, stdout=log, stderr=log)
        chrome = subprocess.Popen([BROWSER, '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', f'--remote-debugging-port={debug_port}', f'--remote-allow-origins=http://localhost:{debug_port}', '--user-data-dir=' + profile, 'about:blank'], stdout=log, stderr=log)
        client = None
        try:
            deadline = time.time() + 20
            while True:
                try:
                    opener.open(f'http://127.0.0.1:{debug_port}/json', timeout=1).close()
                    opener.open(url, timeout=1).close()
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(.1)
            client = CDP(debug_port)
            run(client, url, output)
            print('Browser checks passed. Artifacts:', output)
        finally:
            if client:
                client.close()
            for process in [chrome, server]:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
