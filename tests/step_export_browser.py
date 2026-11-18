"""Browser STEP downloads read independently by the Open CASCADE CAD kernel.

Development only: pip install cadquery-ocp-novtk (no browser runtime dependency).
"""
import os
from pathlib import Path
import tempfile

import browser_smoke as smoke
from model_formats_browser import import_files,convert


def read_step(path):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    reader=STEPCAFControl_Reader();reader.SetNameMode(True)
    assert reader.ReadFile(str(path))==IFSelect_RetDone
    document=TDocStd_Document(TCollection_ExtendedString('STEP check'))
    assert reader.Transfer(document)
    return document,XCAFDoc_DocumentTool.ShapeTool_s(document.Main()),reader


def bounds(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box=Bnd_Box();BRepBndLib.Add_s(shape,box)
    assert not box.IsVoid()
    return box.Get()


def tree(tool):
    from OCP.TDF import TDF_LabelSequence,TDF_Label
    from OCP.TDataStd import TDataStd_Name
    def visit(label):
        name=TDataStd_Name()
        title=name.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),name) else ''
        referred=TDF_Label()
        target=referred if tool.GetReferredShape_s(label,referred) else label
        children=TDF_LabelSequence();tool.GetComponents_s(target,children)
        return {'name':title,'label':label,'definition':target,'children':[visit(children.Value(i)) for i in range(1,children.Length()+1)]}
    roots=TDF_LabelSequence();tool.GetFreeShapes(roots)
    return [visit(roots.Value(i)) for i in range(1,roots.Length()+1)]


def part_bounds(tool,root):
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX
    result={}
    def visit(node,parent):
        location=parent.Multiplied(tool.GetLocation_s(node['label']))
        if not node['children']:
            shape=tool.GetShape_s(node['definition']).Moved(location)
            if not shape.IsNull() and TopExp_Explorer(shape,TopAbs_VERTEX).More():result[node['name']]=bounds(shape)
        for child in node['children']:visit(child,location)
    visit(root,TopLoc_Location());return result


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Network.enable')
    client.call('Page.navigate',{'url':os.environ.get('URDF_STEP_TEST_URL',url)})
    client.until('!!window.urdfWorkbench')
    client.js('''window.formatDownloads=[];window.formatClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.blob()).then(blob=>new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(blob);})).then(data=>formatDownloads.push({name,data}));}else formatClick.call(this)}''')
    # Parent rotation, child translation, visual rotation, mesh scale, a mesh shared
    # by two instances, an empty intermediate link and collision-only fallback.
    obj='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    urdf='''<robot name="装配 'test'">
<link name="base"><visual><origin xyz=".1 .2 .3" rpy="0 0 1.5707963267948966"/><geometry><mesh filename="meshes/tetra.obj" scale=".02 .03 .04"/></geometry></visual></link>
<link name="empty"/>
<link name="arm"><visual><origin xyz="0 0 .1"/><geometry><mesh filename="meshes/tetra.obj" scale=".02 .03 .04"/></geometry></visual><visual><geometry><box size=".1 .2 .3"/></geometry></visual></link>
<link name="tip"><collision><geometry><box size=".02 .04 .06"/></geometry></collision></link>
<joint name="base_to_empty" type="fixed"><parent link="base"/><child link="empty"/><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/></joint>
<joint name="empty_to_arm" type="revolute"><parent link="empty"/><child link="arm"/><origin xyz=".4 .5 .6" rpy=".2 .3 .4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
<joint name="arm_to_tip" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="0 0 .7"/></joint>
</robot>'''
    files={'assembly.urdf':urdf,'meshes/tetra.obj':obj}
    import_files(client,files,"装配 'test'")
    # Display settings / slider changes must not change the source's exported zero pose.
    client.js("urdfWorkbench.viewer.robot.setJointValue('empty_to_arm',.7)")
    package,report=convert(client,'step')
    assert report['units']['length']=='mm' and report['step']['pose']=='urdf-zero'
    assert report['links']==4 and report['joints']==3
    assert report['step']['geometryDefinitions']==3,report
    assert set(package)=={'model.step'}
    path=output/'assembly.step';path.write_bytes(package['model.step'])
    assert package['model.step'].startswith(b'ISO-10303-21;')
    document,tool,reader=read_step(path)
    roots=tree(tool)
    assert len(roots)==1 and roots[0]['name']=="装配 'test'",roots
    # Independently compute expected world-space mesh bounds from source URDF.
    import model_formats as formats
    import numpy as np
    from test_model_formats import poses
    with tempfile.TemporaryDirectory(prefix='step-source-') as folder:
        source=Path(folder)
        for name,value in files.items():
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
        model=formats.read_model(source,source/'assembly.urdf')[0];world=poses(model,{})
        vertices=[];expected_parts={}
        for link in model.links.values():
            for i,geometry in enumerate(link.visuals or link.collisions):
                mesh=formats.shape_mesh(geometry);mesh.apply_transform(world[link.name]@geometry.pose);vertices.extend(mesh.vertices)
                expected_parts[link.name+('/visual_' if link.visuals else '/collision_')+str(i+1)]=mesh.bounds.flatten()*1000
        v=np.array(vertices)*1000
        expected=np.r_[v.min(axis=0),v.max(axis=0)]
        actual=bounds(tool.GetShape_s(roots[0]['label']))
        np.testing.assert_allclose(actual,expected,atol=2e-4)
        actual_parts=part_bounds(tool,roots[0])
        assert set(actual_parts)==set(expected_parts)
        for name in actual_parts:np.testing.assert_allclose(actual_parts[name],expected_parts[name],atol=2e-4)
    def names(node):return [node['name']]+[name for child in node['children'] for name in names(child)]
    all_names=names(roots[0])
    for name in ['base','empty','arm','tip']:assert name in all_names,(name,all_names)
    node=roots[0]
    for name in ['base','empty','arm','tip']:
        node=next(child for child in node['children'] if child['name']==name)
    for joint in ['base_to_empty','empty_to_arm','arm_to_tip']:assert joint in package['model.step'].decode()
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.BRepCheck import BRepCheck_Analyzer
    solid=TopExp_Explorer(tool.GetShape_s(roots[0]['label']),TopAbs_SOLID);count=0
    while solid.More():
        assert BRepCheck_Analyzer(solid.Current()).IsValid()
        count+=1;solid.Next()
    assert count==4,count
    print('PASS CAD kernel reads named nested assembly, shared geometry, valid solids, mm units and accurate zero-pose bounds',flush=True)
    # An open triangle remains an open CAD surface; it must not become a fake solid.
    surface='<robot name="surface"><link name="base"><visual><geometry><mesh filename="open.obj"/></geometry></visual></link></robot>'
    # A near-collinear triangle from a CAD tessellation must not inflate reader
    # tolerance and the bounding box by several millimetres (seen in the Go2 DAE).
    open_mesh='v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 .008 .0000000000000001\nv 0 .016 0\nf 1 2 3\nf 1 4 5\n'
    import_files(client,{'surface.urdf':surface,'open.obj':open_mesh},'surface')
    package,report=convert(client,'step');(output/'surface.step').write_bytes(package['model.step'])
    document,tool,reader=read_step(output/'surface.step');roots=tree(tool)
    shape=tool.GetShape_s(roots[0]['label']);assert not shape.IsNull()
    assert not TopExp_Explorer(shape,TopAbs_SOLID).More()
    assert report['step']['surfaceComponents']==1
    assert report['step']['triangles']==1
    assert any('skipped 1 degenerate' in note for note in report['warnings'])
    np.testing.assert_allclose(bounds(shape),[0,0,0,100,100,0],atol=2e-4)
    print('PASS open meshes remain readable CAD surfaces with explicit conversion notes',flush=True)
    if os.environ.get('URDF_STEP_TEST_GO2')=='1':
        client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
        client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description" && !urdfWorkbench.loading',60)
        expected=client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),r=urdfWorkbench.viewer.robot,out={};r.updateWorldMatrix(true,true);const inverse=r.matrixWorld.clone().invert(),point=new T.Vector3();for(const [name,link] of Object.entries(r.links)){const visuals=link.children.filter(c=>c.isURDFVisual);for(const [i,visual] of visuals.entries()){const box=new T.Box3();visual.traverse(mesh=>{const p=mesh.geometry?.attributes.position;if(!p)return;const matrix=inverse.clone().multiply(mesh.matrixWorld);for(let i=0;i<p.count;i++)box.expandByPoint(point.fromBufferAttribute(p,i).applyMatrix4(matrix));});if(!box.isEmpty())out[name+'/visual_'+(i+1)]=[...box.min.toArray(),...box.max.toArray()].map(v=>v*1000);}}return out})()''')
        link_names=client.js('Object.keys(urdfWorkbench.viewer.robot.links)')
        joint_names=client.js('Object.keys(urdfWorkbench.viewer.robot.joints)')
        package,report=convert(client,'step');path=output/'go2.step';path.write_bytes(package['model.step'])
        assert report['links']==len(link_names) and report['joints']==len(joint_names)
        print('PASS actual Go2 STEP downloads: '+str(len(package['model.step']))+' bytes, '+str(report['step']['triangles'])+' unique triangles',flush=True)
        document,tool,reader=read_step(path);roots=tree(tool);assert len(roots)==1
        actual=part_bounds(tool,roots[0])
        for name in expected:
            assert name in actual,name
            np.testing.assert_allclose(actual[name],expected[name],atol=.002)
        assert all(name in names(roots[0]) for name in link_names)
        print('PASS CAD kernel reads all '+str(len(link_names))+' Go2 links and every visual instance at its precise URDF position',flush=True)
    requests=[e['params']['request']['url'] for e in client.events if e.get('method')=='Network.requestWillBeSent']
    assert not any('/api/formats/' in request for request in requests)
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    print('PASS STEP download runs in the browser without a Python service',flush=True)


if __name__=='__main__':
    os.environ['URDF_TEST_FORMATS']='0';smoke.run=run;smoke.main()
