import numpy as np
import coacd
import trimesh
import xml.etree.ElementTree as ET

mesh_file = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/panda/meshes/collision/objects/chair.stl"

mesh = trimesh.load(mesh_file,force="mesh")
#do mean center
mesh.vertices -= mesh.center_mass


mesh = coacd.Mesh(mesh.vertices,mesh.faces)
parts = coacd.run_coacd(mesh)
scale = "1.3 1.3 1.3"

name = "chair"
numParts = 0

Xmjinc = ET.Element("mujocoinclude")
Xasset = ET.SubElement(Xmjinc,"asset")
XcolM = ET.SubElement(Xasset,"mesh",attrib={"name":"chair_viz","file":"../meshes/visual/objects/chair.stl","scale":scale})

for i,part in enumerate(parts):
    Pmesh = trimesh.Trimesh(part[0],part[1])
    path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/panda/meshes/collision/objects/decompositions/"
    filename = name+"Part-"+str(i)+".stl"
    Pmesh.export(path+filename)
    Xatrdict = {"name":filename.strip(".stl"),"file":"../meshes/collision/objects/decompositions/"+filename, "scale":scale}
    Xmesh = ET.SubElement(Xasset,"mesh",attrib=Xatrdict)
    numParts += 1

XAssetTree = ET.ElementTree(Xmjinc)

with open("/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/panda/assets/chair.xml", "wb") as f:
    XAssetTree.write(f, encoding="utf-8")

print("asset XML file created successfully.")

Xobj = ET.Element("mujocoinclude")
Xbody = ET.SubElement(Xobj,"body",attrib={"name":"collision_object","pos":"0.0 0.3 0.2","quat":"0 0 0 1"})
Xjoint = ET.SubElement(Xbody,"joint",attrib={"type":"free","name":"object_virtual_joint","pos":"0.0 1.0 0.0","damping":"5"})
XvisG = ET.SubElement(Xbody,"geom",attrib={"class":"object_viz","mesh":"chair_viz"})

for i in range(numParts):
    XcolG = ET.SubElement(Xbody,"geom",attrib={"class":"object_col","mesh":"chairPart-"+str(i)})

Xinert = ET.SubElement(Xbody,"inertial",attrib={"pos":"0 0 0","mass":"1","diaginertia":"2 2 2"})


XobTree = ET.ElementTree(Xobj)
with open("/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/object.xml", "wb") as f:
    XobTree.write(f, encoding="utf-8")

print("body XML file created successfully.")
