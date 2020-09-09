import vtk
import os
from ctypes import *
import numpy as np

iren = vtk.vtkRenderWindowInteractor()
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

renWin = vtk.vtkRenderWindow()
iren.SetRenderWindow(renWin)

ren = vtk.vtkRenderer()
renWin.AddRenderer(ren)

def MakeActor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


library_path = os.path.dirname(os.path.abspath(__file__))

lib = cdll.LoadLibrary(os.path.join(library_path, "manifold_lib"))

lib.Calculate.restype = py_object
lib.Simplify.restype = py_object

def CalculateManifold(polydata, resolution = 1000):

    #Get Vertex and Face Array
    nPoints = polydata.GetNumberOfPoints()
    nFaces = polydata.GetNumberOfCells()

    v = []
    f = []

    for i in range(nPoints):
        v.append(polydata.GetPoint(i))

    for i in range(nFaces):
        cell = polydata.GetCell(i)

        cell.GetPointId(0)

        f.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2) ])

    v = np.array(v)
    f = np.array(f)


    print(resolution)

    #Calculate
    calculated = lib.Calculate( v.ctypes.data_as(POINTER(c_double)), v.size,
                    f.ctypes.data_as(POINTER(c_int)), f.size,
                    c_int(resolution))

    vertices = np.array( calculated['vertices'])
    vertices = vertices.reshape(int(vertices.size/3), 3)
    faces = np.array( calculated['faces'])
    faces = faces.reshape(int(faces.size/3), 3)


    result = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(vertices.shape[0])
    for i in range(vertices.shape[0]):
        points.SetPoint(i, vertices[i])
    result.SetPoints(points)

    cells = vtk.vtkCellArray()
    for i in range(faces.shape[0]):
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, faces[i][0])
        cell.GetPointIds().SetId(1, faces[i][1])
        cell.GetPointIds().SetId(2, faces[i][2])
        cells.InsertNextCell(cell)
    result.SetPolys(cells)

    return result


def Simplify(polydata, max_faces):
    
    
    #Get Vertex and Face Array
    nPoints = polydata.GetNumberOfPoints()
    nFaces = polydata.GetNumberOfCells()

    v = []
    f = []

    for i in range(nPoints):
        v.append(polydata.GetPoint(i))

    for i in range(nFaces):
        cell = polydata.GetCell(i)

        cell.GetPointId(0)

        f.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2) ])

    v = np.array(v)
    f = np.array(f)


    calculated = lib.Simplify(v.ctypes.data_as(POINTER(c_double)), v.size,
                    f.ctypes.data_as(POINTER(c_int)), f.size, c_int(max_faces))


    vertices = np.array( calculated['vertices'])
    vertices = vertices.reshape(int(vertices.size/3), 3)
    faces = np.array( calculated['faces'])
    faces = faces.reshape(int(faces.size/3), 3)


    result = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(vertices.shape[0])
    for i in range(vertices.shape[0]):
        points.SetPoint(i, vertices[i])
    result.SetPoints(points)

    cells = vtk.vtkCellArray()
    for i in range(faces.shape[0]):
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, faces[i][0])
        cell.GetPointIds().SetId(1, faces[i][1])
        cell.GetPointIds().SetId(2, faces[i][2])
        cells.InsertNextCell(cell)
    result.SetPolys(cells)

    return result




if __name__ == "__main__":
    data_path = "../exp/sampleData/mandibular_nonmanifold.obj"    


    reader = vtk.vtkOBJReader()
    reader.SetFileName(data_path)
    reader.Update()

    polydata = reader.GetOutput()


    #Calculate Manifold, Save OBJ
    manifold = CalculateManifold(polydata, 100)


    #After Calculating
    actor = MakeActor(manifold)
    ren.AddActor(actor)

    renWin.Render()

    iren.Initialize()
    iren.Start()


    #Run Pro