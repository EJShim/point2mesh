from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class IWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())



        self.mainLayout = QVBoxLayout()
        self.rendererLayout = QVBoxLayout()
        self.buttonLayout = QHBoxLayout()

        self.mainLayout.addLayout(self.rendererLayout)
        self.mainLayout.addLayout(self.buttonLayout)


        self.centralWidget().layout().addLayout(self.mainLayout)


        #Initalize Buttons
        self.buttonLayout.addWidget(QPushButton("Import"))
        self.buttonLayout.addWidget(QPushButton("Start"))
        self.buttonLayout.addWidget(QPushButton("Upsample"))
        self.buttonLayout.addWidget(QPushButton("Save"))

    def SetVTK(self, vtkWidget):
        self.rendererLayout.addWidget(vtkWidget)

    