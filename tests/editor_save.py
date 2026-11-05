"""Source overwrite checks with real Chromium filesystem streams; only native pickers are substituted."""
import json
from model_interaction import rendered_frame

XML='<robot name="save_fixture"><link name="base"><visual><geometry><box size=".2 .2 .2"/></geometry></visual></link></robot>'


def edit(client,text):
    client.js('document.getElementById("source").value='+json.dumps(text)+';document.getElementById("source").dispatchEvent(new Event("input"))')


def save(client):
    client.js('document.getElementById("save").click()')
    client.until('!document.getElementById("save").disabled')
    rendered_frame(client)


def check_editor_save(client,output,verify):
    def check(client,expression,message):
        verify(client,'(async()=>('+expression+'))()',message)
    def script(code):
        client.js('(async()=>{'+code+'})()')
    # Start with an empty editor for the new-file path. OPFS is isolated to this
    # disposable browser profile; no bundled robot source is changed by tests.
    client.call('Page.reload')
    client.until('!!window.urdfWorkbench&&!urdfWorkbench.viewer.robot')
    client.js('''(async()=>{
      const dir=await (await navigator.storage.getDirectory()).getDirectoryHandle('save-tests',{create:true});
      window.saveTest={dir,open:window.showOpenFilePicker,folder:window.showDirectoryPicker,save:window.showSaveFilePicker,openCalls:0,saveCalls:0};
      saveTest.put=async(h,text)=>{const w=await FileSystemFileHandle.prototype.createWritable.call(h);await w.write(text);await w.close()};
      saveTest.main=await dir.getFileHandle('original.urdf',{create:true});
      saveTest.other=await dir.getFileHandle('other.urdf',{create:true});
      saveTest.fresh=await (await navigator.storage.getDirectory()).getFileHandle('fresh.urdf',{create:true});
      window.showSaveFilePicker=async()=>{saveTest.saveCalls++;return saveTest.fresh};
      window.showOpenFilePicker=async()=>{saveTest.openCalls++;return [saveTest.main]};
      window.showDirectoryPicker=async()=>saveTest.dir;
    })()''')
    edit(client,XML)
    save(client)
    check(client,'(await (await saveTest.fresh.getFile()).text())===document.getElementById("source").value&&document.getElementById("save-state").textContent==="已保存 · 未应用"&&!urdfWorkbench.viewer.robot','new editor content saves once to a chosen file without silently reloading the model')
    script('saveTest.original='+json.dumps(XML)+';await saveTest.put(saveTest.main,saveTest.original);await saveTest.put(saveTest.other,saveTest.original.replace("save_fixture","other_fixture"));document.getElementById("load-files").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="save_fixture"&&!urdfWorkbench.loading')
    first=XML.replace('.2 .2 .2','.35 .2 .2')
    edit(client,first)
    client.js('window.beforeSaveRobot=urdfWorkbench.viewer.robot;void 0')
    save(client)
    check(client,'(await (await saveTest.main.getFile()).text())==='+json.dumps(first)+'&&saveTest.openCalls===1&&saveTest.saveCalls===1&&beforeSaveRobot===urdfWorkbench.viewer.robot&&document.getElementById("save-state").textContent==="已保存 · 未应用"&&!document.getElementById("save-state").classList.contains("dirty")','Save overwrites the native-opened original, preserves the loaded model and does not download a copy')
    second=first.replace('.35 .2 .2','.45 .2 .2')
    edit(client,second)
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'s','code':'KeyS','modifiers':2})
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'s','code':'KeyS','modifiers':0})
    client.until('(async()=>((await (await saveTest.main.getFile()).text())==='+json.dumps(second)+'&&!document.getElementById("save").disabled))()')
    check(client,'saveTest.openCalls===1&&saveTest.saveCalls===1','Ctrl+S reuses the exact file handle without another picker')
    client.js('document.getElementById("reload").click()');client.until('!urdfWorkbench.loading')
    check(client,'document.getElementById("save-state").textContent==="已保存"','applying saved code preserves its saved status')
    edit(client,'<robot>');save(client)
    check(client,'!document.getElementById("editor-error").hidden&&(await (await saveTest.main.getFile()).text())==='+json.dumps(second),'invalid XML cannot overwrite the source file')
    third=second.replace('.45 .2 .2','.55 .2 .2');edit(client,third)
    client.js('saveTest.main.queryPermission=async()=>"prompt";saveTest.main.requestPermission=async()=>"denied"')
    save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("写入权限")&&(await (await saveTest.main.getFile()).text())==='+json.dumps(second)+'&&document.getElementById("save-state").classList.contains("dirty")','denying write permission preserves both source contents and unsaved edits')
    client.js('delete saveTest.main.queryPermission;delete saveTest.main.requestPermission;saveTest.main.createWritable=async()=>{const stream=await FileSystemFileHandle.prototype.createWritable.call(saveTest.main);return {write:async text=>{await stream.write(text);throw Error("simulated write failure")},close:()=>stream.close(),abort:()=>stream.abort()}}')
    save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("simulated write failure")&&(await (await saveTest.main.getFile()).text())==='+json.dumps(second)+'&&document.getElementById("save-state").classList.contains("dirty")','a failed write aborts the real file stream and never reports success')
    script('delete saveTest.main.createWritable;await saveTest.put(saveTest.main,saveTest.original+"<!-- external edit -->")')
    save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("其他地方修改")&&(await (await saveTest.main.getFile()).text()).endsWith("<!-- external edit -->")','external file changes are detected before an overwrite')
    script('await saveTest.put(saveTest.main,'+json.dumps(second)+');saveTest.main.createWritable=async()=>{const s=await FileSystemFileHandle.prototype.createWritable.call(saveTest.main);return {write:async text=>{await new Promise(resolve=>saveTest.release=resolve);await s.write(text)},close:()=>s.close(),abort:()=>s.abort()}};document.getElementById("save").click()')
    client.until('!!saveTest.release')
    fourth=third.replace('.55 .2 .2','.65 .2 .2');edit(client,fourth)
    check(client,'document.getElementById("save").disabled&&document.getElementById("reload").disabled','saving locks duplicate writes and model reloads while leaving the editor usable')
    client.js('saveTest.release()');client.until('!document.getElementById("save").disabled')
    check(client,'(await (await saveTest.main.getFile()).text())==='+json.dumps(third)+'&&document.getElementById("source").value==='+json.dumps(fourth)+'&&document.getElementById("save-state").classList.contains("dirty")','typing during a save keeps newer text unsaved instead of losing it')
    client.js('delete saveTest.main.createWritable');save(client)
    # A native directory supplies one distinct retained handle for each source.
    client.js('window.beforeFolderRobot=urdfWorkbench.viewer.robot;document.getElementById("load-folder").click()');client.until('!!document.querySelector("dialog[open]")')
    client.js('[...document.querySelector("dialog[open]").querySelectorAll("button")].find(b=>b.textContent.endsWith("original.urdf")).click()')
    client.until('urdfWorkbench.viewer.robot!==beforeFolderRobot&&urdfWorkbench.viewer.robot?.robotName==="save_fixture"&&!urdfWorkbench.loading')
    fifth=fourth.replace('.65 .2 .2','.75 .2 .2');edit(client,fifth);save(client)
    client.js('document.querySelector(`#file-tree button[data-path="save-tests/other.urdf"]`).click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="other_fixture"&&!urdfWorkbench.loading')
    client.js('document.querySelector(`#file-tree button[data-path="save-tests/original.urdf"]`).click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="save_fixture"&&!urdfWorkbench.loading')
    check(client,'document.getElementById("source").value==='+json.dumps(fifth)+'&&(await (await saveTest.other.getFile()).text()).includes("other_fixture")&&saveTest.openCalls===1','directory imports save only the active source and read updated text when returning through the file tree')
    # The legacy File-input path has no handle. Its first Save must bind the
    # matching original; subsequent saves must reuse that granted identity.
    legacy='''(async()=>{window.beforeLegacyRobot=urdfWorkbench.viewer.robot;const d=new DataTransfer();d.items.add(await saveTest.main.getFile());const i=document.getElementById('file-input');i.files=d.files;i.dispatchEvent(new Event('change'))})()'''
    client.js(legacy);client.until('urdfWorkbench.viewer.robot!==beforeLegacyRobot&&!urdfWorkbench.loading')
    sixth=fifth.replace('.75 .2 .2','.85 .2 .2');edit(client,sixth)
    client.js('window.showOpenFilePicker=async()=>{saveTest.openCalls++;return [saveTest.other]}');save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("original.urdf")&&(await (await saveTest.other.getFile()).text()).includes("other_fixture")','binding a legacy import rejects a different source filename')
    script('saveTest.wrong=await (await saveTest.dir.getDirectoryHandle("wrong-folder",{create:true})).getFileHandle("original.urdf",{create:true});await saveTest.put(saveTest.wrong,"unrelated source");window.showOpenFilePicker=async()=>[saveTest.wrong]')
    save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("其他地方修改")&&(await (await saveTest.wrong.getFile()).text())==="unrelated source"','binding also rejects a same-name file with different original contents')
    client.js('window.showOpenFilePicker=async()=>{throw new DOMException("cancelled","AbortError")}');save(client)
    check(client,'document.getElementById("editor-error").hidden&&document.getElementById("save-state").classList.contains("dirty")&&(await (await saveTest.main.getFile()).text())==='+json.dumps(fifth),'cancelling the source picker retains unsaved text and leaves the original untouched')
    client.js('window.showOpenFilePicker=undefined');save(client)
    check(client,'document.getElementById("editor-error").textContent.includes("Chrome / Edge")&&document.getElementById("save-state").classList.contains("dirty")','unsupported browsers report the missing overwrite capability instead of pretending a download is a save')
    client.js('window.showOpenFilePicker=async()=>{saveTest.openCalls++;return [saveTest.main]}');save(client)
    check(client,'(await (await saveTest.main.getFile()).text())==='+json.dumps(sixth)+'&&!document.getElementById("save-state").classList.contains("dirty")','a legacy import binds its matching original and writes edited source successfully')
    count=client.js('saveTest.openCalls');edit(client,sixth+'<!-- second save -->');save(client)
    check(client,'saveTest.openCalls==='+str(count)+'&&(await (await saveTest.main.getFile()).text()).endsWith("<!-- second save -->")','the next save of a bound legacy import directly overwrites without reselecting its source')
    script('window.beforeDropRobot=urdfWorkbench.viewer.robot;const d=new DataTransfer();d.items.add(await saveTest.main.getFile());const native=DataTransferItem.prototype.getAsFileSystemHandle;DataTransferItem.prototype.getAsFileSystemHandle=async()=>saveTest.main;try{window.dispatchEvent(new DragEvent("drop",{dataTransfer:d,cancelable:true}))}finally{DataTransferItem.prototype.getAsFileSystemHandle=native}')
    client.until('urdfWorkbench.viewer.robot!==beforeDropRobot&&!urdfWorkbench.loading')
    edit(client,sixth+'<!-- dropped source -->');save(client)
    check(client,'saveTest.openCalls==='+str(count)+'&&(await (await saveTest.main.getFile()).text()).endsWith("<!-- dropped source -->")','modern drag and drop retains its writable source handle')
    client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55)
    script('saveTest.exampleText=document.getElementById("source").value;saveTest.example=await saveTest.dir.getFileHandle("go2.urdf",{create:true});await saveTest.put(saveTest.example,saveTest.exampleText);window.showOpenFilePicker=async()=>{saveTest.openCalls++;return [saveTest.example]}')
    example=client.js('saveTest.exampleText')
    edit(client,example+'\n<!-- saved example -->');save(client)
    check(client,'saveTest.openCalls==='+str(count+1)+'&&(await (await saveTest.example.getFile()).text()).endsWith("<!-- saved example -->")&&document.getElementById("save-state").textContent==="已保存 · 未应用"','HTTP-loaded examples bind a matching source on first save and write back successfully')
    edit(client,example+'\n<!-- saved again -->');save(client)
    check(client,'saveTest.openCalls==='+str(count+1)+'&&(await (await saveTest.example.getFile()).text()).endsWith("<!-- saved again -->")','HTTP-loaded examples reuse their granted handle on later saves')
    client.screenshot(str(output/'editor-saved-source.png'))
    client.js('window.showOpenFilePicker=saveTest.open;window.showDirectoryPicker=saveTest.folder;window.showSaveFilePicker=saveTest.save')
