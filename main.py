import os, sys
import time
import vtk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from gui import IWindow
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from pymanifold import CalculateManifold
root = os.path.dirname(os.path.abspath(__file__))


import torch
from p2m.models.layers.mesh import Mesh, vtkMesh, PartMesh
from p2m.models.networks import init_net, sample_surface, local_nonuniform_penalty
from p2m import utils
import numpy as np
from p2m.models.losses import chamfer_distance, BeamGapLoss
from p2m.options import Options

app = QApplication([])

options = Options()
opts = options.args

opts.input_pc = "./data/sample1.vtp"
opts.initial_mesh = "./data/manifold.obj"
opts.iterations = 6000
opts.upsamp = 1000
opts.lr = 1.1e-5


torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))


iren = QVTKRenderWindowInteractor()
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
# #Initilaize Renderer
ren = vtk.vtkRenderer()
renWin = iren.GetRenderWindow()
renWin.AddRenderer(ren)



# def ASDF(polydata):
    
#     glyph = vtk.vtkVertexGlyphFilter()
#     glyph.SetInputData(targetPoly)
#     glyph.Update()


#     sphereSource = vtk.vtkSphereSource()
#     sphereSource.SetRadius(10)
#     sphereSource.SetPhiResolution(100)
#     sphereSource.SetThetaResolution(100)
#     sphereSource.Update()


#     smoothFilter = vtk.vtkSmoothPolyDataFilter()
#     smoothFilter.SetInputConnection(0, sphereSource.GetOutputPort())
#     smoothFilter.SetInputData(1, glyph.GetOutput())
#     smoothFilter.Update()

#     convexHull = smoothFilter.GetOutput()



#     return convexHull




def UpdateGT(polydata, vs):
    for idx, pos in enumerate(vs):
        polydata.GetPoints().SetPoint(idx, pos[0], pos[1], pos[2])

    polydata.GetPoints().Modified()

def MakeInitMeshActor(polydata):
    mapper = vtk.vtkPolyDataMapper()    
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.1, .4, .9)
    

    return actor

def MakePointCloudActor(polydata):

    bounds = polydata.GetBounds()
    x1 = np.array([bounds[0], bounds[2], bounds[4]])
    x2 = np.array([bounds[1], bounds[3], bounds[5]])
    d = np.linalg.norm(x1-x2)

    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetRadius(d/1000)    
    mapper.SetInputData(polydata)


    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.9, .4, .1)

    return actor


class Worker(QThread):

    backwarded = pyqtSignal(object)
    upsampled = pyqtSignal(object)

    def __init__(self, initPoly, targetPoly):
        super().__init__()

        self.initPoly = initPoly
        self.targetPoly = targetPoly

    
    def MakeInputData(self, pointcloud):
        xyz = []
        normal = []

        nPoints = pointcloud.GetNumberOfPoints()

        for pid in range(nPoints):
            xyz.append(pointcloud.GetPoint(pid))
            normal.append(pointcloud.GetPointData().GetNormals().GetTuple(pid))

        return np.array(xyz), np.array(normal)


    def run(self):
          
        mesh = vtkMesh(self.initPoly, device=device, hold_history=True)
        
        # input point cloud
        input_xyz, input_normals = self.MakeInputData(self.targetPoly)
        # normalize point cloud based on initial mesh
        input_xyz /= mesh.scale
        input_xyz += mesh.translations[None, :]
        input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
        input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

        part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
        print(f'number of parts {part_mesh.n_submeshes}')
        net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)

        beamgap_loss = BeamGapLoss(device)

        if opts.beamgap_iterations > 0:
            print('beamgap on')
            beamgap_loss.update_pm(part_mesh, torch.cat([input_xyz, input_normals], dim=-1))

        for i in range(opts.iterations):
            num_samples = options.get_num_samples(i % opts.upsamp)
            if opts.global_step:
                optimizer.zero_grad()
            start_time = time.time()
            for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
                if not opts.global_step:
                    optimizer.zero_grad()
                part_mesh.update_verts(est_verts[0], part_i)
                num_samples = options.get_num_samples(i % opts.upsamp)
                recon_xyz, recon_normals = sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0), num_samples)
                # calc chamfer loss w/ normals
                recon_xyz, recon_normals = recon_xyz.type(options.dtype()), recon_normals.type(options.dtype())
                xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, x_normals=recon_normals, y_normals=input_normals,unoriented=opts.unoriented)

                if (i < opts.beamgap_iterations) and (i % opts.beamgap_modulo == 0):
                    loss = beamgap_loss(part_mesh, part_i)
                else:
                    loss = (xyz_chamfer_loss + (opts.ang_wt * normals_chamfer_loss))
                if opts.local_non_uniform > 0:
                    loss += opts.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()
                loss.backward()
                if not opts.global_step:
                    optimizer.step()
                    scheduler.step()
                part_mesh.main_mesh.vs.detach_()
            if opts.global_step:
                optimizer.step()
                scheduler.step()
            end_time = time.time()

            
            UpdateGT(self.initPoly, mesh.export())
            obj = {
                "vs" : mesh.export(),
                "log" : f'iter: {i} out of: {opts.iterations}; loss: {loss.item():.4f};' f' sample count: {num_samples}; time: {end_time - start_time:.2f}' 
            }

            self.backwarded.emit(obj)


            if (i > 0 and (i + 1) % opts.upsamp == 0):
                mesh = part_mesh.main_mesh
                num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))

                if num_faces > len(mesh.faces) or opts.manifold_always:

                    self.initPoly = utils.vtk_upsample(self.initPoly , res=num_faces)
                    mesh = vtkMesh(self.initPoly, device=device, hold_history=True)                    
                    
                    part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
                    print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
                    net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
                    if i < opts.beamgap_iterations:
                        print('beamgap updated')
                        beamgap_loss.update_pm(part_mesh, input_xyz)

                    self.upsampled.emit(self.initPoly)



def loadTemplateSphere(polydata = None):
    reader = vtk.vtkSTLReader()
    reader.SetFileName("./exp/templatesphere.stl")
    reader.Update()

    result = reader.GetOutput()
    
    if polydata:
        center = polydata.GetCenter()
        bounds = polydata.GetBounds()

        polyRange = np.array( [ bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4] ])
        scale = np.max(polyRange)/2
        
        transform = vtk.vtkTransform()
        transform.Translate(center[0], center[1], center[2])
        transform.Scale(polyRange[0], polyRange[1], polyRange[2])
        transform.Update()

        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetInputData(result)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        result = transformFilter.GetOutput()


            
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(polydata)
        glyph.Update()



        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(0, result)
        smoothFilter.SetInputData(1, glyph.GetOutput())
        smoothFilter.Update()

        smoothFilter2 = vtk.vtkSmoothPolyDataFilter()
        smoothFilter2.SetInputData(0, result)
        smoothFilter2.SetInputData(1, smoothFilter.GetOutput())
        smoothFilter2.SetRelaxationFactor(1)
        smoothFilter2.Update()

        result = smoothFilter2.GetOutput()
    
    return result

def MakeInitMesh(pcPoly):


    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(targetPoly.GetCenter())
    sphereSource.SetRadius(rad)
    sphereSource.SetThetaResolution(40)
    sphereSource.SetPhiResolution(40)
    # sphereSource.Update()
    
    cleanPoly = vtk.vtkCleanPolyData()
    cleanPoly.SetInputConnection(sphereSource.GetOutputPort())
    cleanPoly.Update()
    initMesh = cleanPoly.GetOutput()


    print(initMesh.GetNumberOfPoints())
    initMesh.GetPointData().RemoveArray("Normals")

    
    return initMesh


if __name__ == "__main__":

    # initReader = vtk.vtkOBJReader()
    # initReader.SetFileName(os.path.join( opts.initial_mesh))
    # initReader.Update()
    # initMesh = initReader.GetOutput()

    # cleanPoly = vtk.vtkCleanPolyData()
    # cleanPoly.SetInputData(initMesh)
    # cleanPoly.Update()
    # initMesh = cleanPoly.GetOutput()

    # initMesh.GetPointData().RemoveArray("Normals")


    # print(initMesh.GetNumberOfPoints())



    if opts.input_pc[-3:] == "ply":
        targetReader = vtk.vtkPLYReader()
    elif opts.input_pc[-3:] == "vtp":
        targetReader = vtk.vtkXMLPolyDataReader()

    targetReader.SetFileName(os.path.join( opts.input_pc))
    targetReader.Update()

    targetPoly = targetReader.GetOutput()
    if not targetPoly.GetPointData().GetNormals():
        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputData(targetReader.GetOutput())
        normalGenerator.Update()
        targetPoly = normalGenerator.GetOutput()

    # initMesh = ASDF(targetPoly)
    #Get Target Poly Radius
    bounds = targetPoly.GetBounds()
    x1 = np.array([bounds[0], bounds[2], bounds[4]])
    x2 = np.array([bounds[1], bounds[3], bounds[5]])
    rad = np.linalg.norm(x1-x2) / 4


    #Calculate InitMesh
    initMesh = CalculateManifold(targetPoly, 100)
    


    # initMesh = loadTemplateSphere(targetPoly)
    

    initMeshActor =  MakeInitMeshActor(initMesh)
    pcActor = MakePointCloudActor(targetPoly)

    ren.AddActor(initMeshActor)
    # ren.AddActor(pcActor)

    logActor = vtk.vtkTextActor()
    logActor.SetInput("Log")
    logActor.SetPosition2(100, 100)
    logActor.GetTextProperty().SetFontSize(20)
    logActor.GetTextProperty().SetColor(0, 1,0)
    ren.AddActor2D(logActor)

    renWin.Render()
    

    #Run Thread
    def test(obj):
        # UpdateGT(initMesh, obj["vs"])
        logActor.SetInput(obj["log"])
        renWin.Render()

    def upsampled(polydata):
        initMeshActor.GetMapper().SetInputData(polydata)
        renWin.Render()

    trainingWorker = Worker(initMesh, targetPoly)
    trainingWorker.backwarded.connect(test)
    trainingWorker.upsampled.connect(upsampled)
    trainingWorker.start()

    window = IWindow()
    window.SetVTK(iren)
    window.show()
    sys.exit(app.exec_())

    exit()