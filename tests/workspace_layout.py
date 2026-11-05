"""Dock layout checks: real tab, resize and collapse interactions keep 3D space usable."""
from model_interaction import pointer, rendered_frame


def click(client, element_id):
    client.js(f'document.getElementById("{element_id}").click()')
    rendered_frame(client)


def check_workspace_layout(client, output, check):
    check(client,'["files","joints","structure"].every(n=>document.getElementById(n+"-panel").parentElement.id==="workspace-content")&&document.getElementById("editor-panel").parentElement===document.body','files, joints and structure share one left dock; editor is independent')
    check(client,'(()=>{const l=document.getElementById("workspace-sidebar").getBoundingClientRect(),r=document.getElementById("editor-panel").getBoundingClientRect();return l.left===12&&r.right===innerWidth-12&&Math.abs(l.top-r.top)<1&&r.height>innerHeight*.85&&r.left-l.right>innerWidth*.57})()','full-height sidebars leave over 57% of desktop width for the 3D model')
    client.screenshot(str(output/'workspace-files.png'))
    for name in ['joints','structure','files']:
        click(client,'tab-'+name)
        check(client,f'[...document.querySelectorAll("#workspace-content>.panel")].filter(p=>!p.hidden).length===1&&!document.getElementById("{name}-panel").hidden&&document.getElementById("tab-{name}").getAttribute("aria-selected")==="true"','left tabs show only '+name+' without overlapping panels')
        if name=='structure':
            check(client,'(()=>{const r=document.querySelector(".structure-node[data-model-root=true] rect").getBoundingClientRect(),h=document.getElementById("tree").getBoundingClientRect();return r.width>0&&r.left>=h.left&&r.right<=h.right&&r.top>=h.top&&r.bottom<=h.bottom})()','structure graph fits its new dock after switching tabs')
            client.screenshot(str(output/'workspace-structure.png'))
        if name=='joints':client.screenshot(str(output/'workspace-joints.png'))
    client.js('document.getElementById("tab-files").focus()')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'ArrowRight','code':'ArrowRight','windowsVirtualKeyCode':39})
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'ArrowRight','code':'ArrowRight','windowsVirtualKeyCode':39})
    check(client,'!document.getElementById("joints-panel").hidden&&document.activeElement.id==="tab-joints"','arrow keys switch tabs and move focus')
    click(client,'sidebar-toggle')
    check(client,'document.getElementById("workspace-sidebar").getBoundingClientRect().width===44&&document.getElementById("joints-panel").hidden','collapsing the workspace leaves a 44px tab rail')
    click(client,'tab-joints')
    check(client,'!document.getElementById("joints-panel").hidden&&!document.getElementById("workspace-sidebar").classList.contains("collapsed")','clicking a collapsed tab restores its content')
    before=client.js('document.getElementById("workspace-sidebar").getBoundingClientRect().width')
    edge=client.js('(()=>{const r=document.getElementById("sidebar-resize").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    pointer(client,'mousePressed',edge,clickCount=1)
    pointer(client,'mouseMoved',{'x':edge['x']+40,'y':edge['y']},buttons=1)
    pointer(client,'mouseReleased',{'x':edge['x']+40,'y':edge['y']},clickCount=1)
    check(client,f'Math.abs(document.getElementById("workspace-sidebar").getBoundingClientRect().width-{before}-40)<1','dragging the inner left edge changes the whole workspace width')
    client.js('document.getElementById("editor-resize").focus()')
    before=client.js('document.getElementById("editor-panel").getBoundingClientRect().width')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'ArrowLeft','code':'ArrowLeft','windowsVirtualKeyCode':37})
    check(client,f'Math.abs(document.getElementById("editor-panel").getBoundingClientRect().width-{before}-16)<1','editor resize handle supports keyboard width adjustment')
    client.js('window.dockOffsetBefore=urdfWorkbench.viewer.camera.view.offsetX')
    click(client,'workspace-toggle')
    client.js('document.querySelector("[data-panel=editor]").click()')
    check(client,'document.getElementById("editor-panel").hidden&&urdfWorkbench.viewer.camera.view.offsetX!==dockOffsetBefore','closing sidebars recenters the camera in the expanded viewport')
    client.screenshot(str(output/'workspace-focus.png'))
    client.js('document.getElementById("restore-layout").click()')
    click(client,'tab-joints')
    for width,height in [(1280,800),(800,900),(390,844)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
        rendered_frame(client)
        check(client,'document.body.scrollWidth===innerWidth&&(()=>{const h=document.querySelector("#joints-panel .panel-header").getBoundingClientRect();return [...document.querySelector("#joints-panel .panel-header").children].filter(c=>c.offsetWidth).every(c=>c.getBoundingClientRect().right<=h.right+1)})()',f'workspace controls fit without overflow at {width}px')
        if width<=1000:check(client,'document.getElementById("editor-panel").hidden','compact layout starts with editor hidden')
        if width==390:
            client.js('document.querySelector("[data-panel=editor]").click()')
            check(client,'!document.getElementById("editor-panel").hidden&&document.getElementById("workspace-sidebar").hidden','phone switches from workspace to editor')
            click(client,'workspace-toggle')
            check(client,'!document.getElementById("workspace-sidebar").hidden&&document.getElementById("editor-panel").hidden&&!document.getElementById("joints-panel").hidden','phone restores the last workspace tab')
        client.screenshot(str(output/f'workspace-{width}.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1866,'height':961,'deviceScaleFactor':1,'mobile':False})
    rendered_frame(client)
    client.js('document.getElementById("restore-layout").click()')
    click(client,'tab-structure')
