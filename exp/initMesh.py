import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root) 
import vtk
from scipy.spatial import ConvexHull
import numpy as np

#Initilaize Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(1000, 1000)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())


def MakeInitMeshActor(polydata):

    mapper = vtk.vtkPolyDataMapper()    
    mapper.SetInputData(polydata)
    

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.1, .4, .9)
    

    return actor

def MakePointCloudActor(polydata):

    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetRadius(.01)    
    mapper.SetInputData(polydata)
    

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.9, .4, .1)

    return actor
def MakeConvexHull(polydata):
    
    #Convex Hull
    nPoints = targetPoly.GetNumberOfPoints()

    vs = []
    for pid in range(nPoints):
        vs.append(targetPoly.GetPoint(pid))
    vs = np.array(vs)
    print(vs.shape)
    hull = ConvexHull(vs)

    convexHull = vtk.vtkPolyData()
    convexHull.SetPoints(targetPoly.GetPoints())
    cells = vtk.vtkCellArray()
    for s in hull.simplices:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, s[0])
        triangle.GetPointIds().SetId(1, s[1])
        triangle.GetPointIds().SetId(2, s[2])
        cells.InsertNextCell(triangle)
    convexHull.SetPolys(cells)


    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(convexHull)
    cleaner.Update()
    convexHull = cleaner.GetOutput()

    return convexHull

def pointSampler(polydata):

    polydata.GetPointData().RemoveArray("Normals")
    
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(targetPoly)
    glyph.Update()


    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetRadius(10)
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)
    sphereSource.Update()


    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputConnection(0, sphereSource.GetOutputPort())
    smoothFilter.SetInputData(1, glyph.GetOutput())    
    smoothFilter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(smoothFilter.GetOutput())
    cleaner.Update()


    convexHull = cleaner.GetOutput()

    print(convexHull.GetNumberOfPoints())

    
    #Apply sampler

    bounds = convexHull.GetBounds()
    polyRange = [ bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4] ]


    sampler = vtk.vtkPolyDataPointSampler()
    sampler.SetInputData(convexHull)
    sampler.SetDistance(polyRange[0]/20)
    sampler.Update()


    smoothFilter2 = vtk.vtkSmoothPolyDataFilter()
    smoothFilter2.SetInputConnection(0, sphereSource.GetOutputPort())
    smoothFilter2.SetInputData(1, sampler.GetOutput())
    smoothFilter2.Update()

    
    convexHull = sampler.GetOutput()

    print(convexHull.GetNumberOfPoints())

    return convexHull





def ASDF(polydata):

    polydata.GetPointData().RemoveArray("Normals")
    
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(targetPoly)
    glyph.Update()


    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetRadius(10)
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)
    sphereSource.Update()


    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(0, sphereSource.GetOutput())
    smoothFilter.SetInputData(1, glyph.GetOutput())    
    smoothFilter.Update()

    return smoothFilter.GetOutput()




    smoothFilter2 = vtk.vtkSmoothPolyDataFilter()
    smoothFilter2.SetInputConnection(0, sphereSource.GetOutputPort())
    smoothFilter2.SetInputData(1, smoothFilter.GetOutput())
    smoothFilter2.Update()
    convexHull = smoothFilter2.GetOutput()

    cleanPoly = vtk.vtkCleanPolyData()
    cleanPoly.SetInputData(convexHull)
    cleanPoly.Update()

    convexHull = cleanPoly.GetOutput()



    return convexHull

def loadTemplateSphere(polydata = None):
    reader = vtk.vtkSTLReader()
    reader.SetFileName("./templatesphere.stl")
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

if __name__ == "__main__":
    
    
    data = sorted(os.listdir(os.path.join(root, "data")))


    targetPoly = None
    initMesh = None


    for filename in data:
        ext = filename[-3:]
        
        if ext == "obj":
            reader = vtk.vtkOBJReader()
        elif ext == "ply":
            reader = vtk.vtkPLYReader()
    
        reader.SetFileName(os.path.join(root, "data", filename))
        reader.Update()

        polydata = reader.GetOutput()

        if filename == "triceratops.ply" : targetPoly = polydata
        if filename == "triceratops_initmesh.obj" : initMesh = polydata



    targetPoly.GetPointData().RemoveArray("Normals")


    initMesh = loadTemplateSphere(targetPoly)

    print(initMesh.GetNumberOfPoints(), initMesh.GetNumberOfCells())

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(initMesh)
    cleaner.Update()
    initMesh = cleaner.GetOutput()

    print(initMesh.GetNumberOfPoints(), initMesh.GetNumberOfCells())


    targetActor = MakePointCloudActor(targetPoly)
    initActor = MakeInitMeshActor(initMesh)
    
    
    


    ren.AddActor(targetActor)
    ren.AddActor(initActor)
    renWin.Render()

    iren.Initialize()
    iren.Start()
