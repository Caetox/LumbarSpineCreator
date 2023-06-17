# install modules
pip_modules = ['scipy', 'vtk', 'qt', 'ctk', 'numpy', 'csv']
for module_ in pip_modules:
    try:
        module_obj = __import__(module_)
    except ImportError:
        logging.info("{0} was not found.\n Attempting to install {0} . . ."
                     .format(module_))
        pip_main(['install', module_])
        
# imports
import os
import vtk
import qt
import ctk
import slicer
import csv
import pandas as pd
from slicer.ScriptedLoadableModule import *
import numpy as np
from scipy.spatial import distance

#
# Create spine from individual sawbones, with input parameters lumbar lordosis angle and minimal and maximal IVD height
#


class LumbarSpineCreator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Lumbar Spine Creator"
        self.parent.categories = ["VisSimTools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Lara Blomenkamp"]
        self.parent.helpText = """
        Lumbar Spine Creator generates an artificial model of the lumbar spine, including vertebrae L1-L5 and the sacrum.
        The user can set the lumbar lordosis angle and a minimum and maximum value for the intervertebral disc height.
        Individual FSU parameters (angles and IVD height) are calculated based on a distribution, which depends on the selected subject position.
        Optionally, the respective parameters can be set for each FSU individually.
        The output includes the models of L1, L2, L3, L4, L5, SA, aligned as a coherent artificial lumbar spine,
        along with a table of measurements about the lumbar spine.
        """
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """University of Koblenz"""  # replace with organization, grant and thanks.

#
# LumbarSpineCreatorWidget
#

class LumbarSpineCreatorWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        scriptPath = os.path.dirname(os.path.abspath(__file__))
        self.outputDirectory = os.path.join(scriptPath, "Output")

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # sawbone directory
        self.sawboneDirectory = os.path.join(scriptPath, "LumbarSpine")

        self.modeSelector = qt.QComboBox()
        self.modeSelector.addItems(["Standing","Supine"])
        self.modeSelector.setCurrentIndex(0)

        #
        # parameter input boxes
        #
        #                       angle       ivdhLmin    ivdhLmax
        values        =  [      55,         7.0,        10.55       ]
        singleSteps   =  [      1,          0.1,         0.1        ]
        rangeMin      =  [      0,          0.1,         0.1        ]
        rangeMax      =  [      100,         30,         50         ]
        prefix        =  [      "",         "Min ",     "Max "      ]
        suffix        =  [      " \u00B0",  " mm",      " mm"       ]

        self.LLangle = qt.QDoubleSpinBox(parametersCollapsibleButton)
        self.ivdhLumbarMin = qt.QDoubleSpinBox(parametersCollapsibleButton)
        self.ivdhLumbarMax = qt.QDoubleSpinBox(parametersCollapsibleButton)

        parameterInputBoxes = [ self.LLangle, self.ivdhLumbarMin, self.ivdhLumbarMax ]
        for p in range(0,len(parameterInputBoxes)):
            parameterInputBox = parameterInputBoxes[p]
            parameterInputBox.setRange(rangeMin[p], rangeMax[p])
            parameterInputBox.setSingleStep(singleSteps[p])
            parameterInputBox.setValue(values[p])
            parameterInputBox.setPrefix(prefix[p])
            parameterInputBox.setSuffix(suffix[p])


        # parameters grid layout
        vLayout = qt.QGridLayout()
        parametersFormLayout.addRow(vLayout)
        vLayout.addWidget(qt.QLabel("Subject Position: "),6,0)
        vLayout.addWidget(self.modeSelector,6,1,1,-1)
        vLayout.addWidget(qt.QLabel("Lordosis Angle:"),7,0)
        vLayout.addWidget(self.LLangle,7,1,1,-1)
        vLayout.addWidget(qt.QLabel("Intervertebral Disc Height:"),8,0)
        vLayout.addWidget(self.ivdhLumbarMin,8,1)
        vLayout.addWidget(self.ivdhLumbarMax,8,2)

        # Create Spine Curves Button
        self.createSpineButton = qt.QPushButton("Create Spine")
        self.createSpineButton.enabled = True
        self.createSpineButton.setStyleSheet("QPushButton{ background-color: Blue }")
        parametersFormLayout.addRow(self.createSpineButton)
        self.createSpineButton.connect('clicked(bool)', self.onCreateSpineButton)

        # individual FSU Angle Parameters
        fsuAnglesCollapsibleButton = ctk.ctkCollapsibleButton()
        fsuAnglesCollapsibleButton.text = "Set Individual FSU Parameters"
        fsuAnglesCollapsibleButton.collapsed=True
        self.layout.addWidget(fsuAnglesCollapsibleButton)

        # Layout within the dummy collapsible button
        fsuAnglesFormLayout = qt.QFormLayout(fsuAnglesCollapsibleButton)

        # param matrices
        self.indivParamMatrix = qt.QTableWidget(2,5)
        fsuAnglesFormLayout.addRow(self.indivParamMatrix)

        labelsL = ["L1-L2", "L2-L3", "L3-L4","L4-L5", "L5-S1"]

        self.indivParamMatrix.setHorizontalHeaderLabels(labelsL)
        self.indivParamMatrix.setVerticalHeaderLabels(["Cobb\nAngle", "Intervertebral\nDisc Height"])
        self.indivParamMatrix.horizontalHeader().setMinimumSectionSize(0)
        self.indivParamMatrix.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        for r in range(0, self.indivParamMatrix.rowCount):
            for c in range (0, self.indivParamMatrix.columnCount):
                paraEdit = qt.QLineEdit("")
                self.indivParamMatrix.setCellWidget(r, c, paraEdit)
                paraEdit.editingFinished.connect(self.onEditIndivParam)


        # connections
        self.modeSelector.currentIndexChanged.connect(self.onEditParam)
        self.parameterWidgets = [self.LLangle, self.ivdhLumbarMin, self.ivdhLumbarMax]
        for parameterWidget in self.parameterWidgets:
            parameterWidget.valueChanged.connect(self.onEditParam)


        self.params = self.calcParams(0, self.parameterWidgets, self.modeSelector.currentIndex)

        # output
        outputCollapsibleButton = ctk.ctkCollapsibleButton()
        outputCollapsibleButton.text = "Output Measurements"
        outputCollapsibleButton.collapsed=True
        self.layout.addWidget(outputCollapsibleButton)
        outputFormLayout = qt.QFormLayout(outputCollapsibleButton)

        # param matrices
        self.outputMatrix = qt.QTableWidget(7,5)
        self.outputMatrix.setHorizontalHeaderLabels(labelsL)
        self.outputMatrix.setVerticalHeaderLabels(["FSU\nAngle", "Wedge\nAngle", "IVD Height\n(Middle)", "IVD Height\n(Anterior)", "IVD Height\n(Posterior)", "IVD Height\n(Avg)", "Lumbar\nLordosis\nAngle"])
        self.outputMatrix.horizontalHeader().setMinimumSectionSize(0)
        self.outputMatrix.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)

        outputFormLayout.addRow(self.outputMatrix)
        
        # Add vertical spacer
        self.layout.addStretch(1)
    
    # enddef setup

    def onEditParam(self):
        parameterWidgets = self.parameterWidgets
        self.params = self.calcParams(0, parameterWidgets, self.modeSelector.currentIndex)

    def onEditIndivParam(self):
        parameterWidgets = self.parameterWidgets
        self.params = self.calcParams(1, parameterWidgets, self.modeSelector.currentIndex)

    def cleanup(self):
        pass

    def calcParams(self, edit, parameterWidgets, mode):

        llAngleWidget = parameterWidgets[0]
        ivdhLumbarMinWidget = parameterWidgets[1]
        ivdhLumbarMaxWidget = parameterWidgets[2]
        inputValues = []

        vtAngle = np.negative(np.divide(llAngleWidget.value,2))
        vtAngles = [vtAngle]
        
        if (edit==0): # spine parameters changed -> calculate FSU parameters

            # average data                   S1-L5,    L5-L4,  L4-L3,  L3-L2,  L2-L1
            # fsu angles
            fsuAnglesSupine             =   [25.8,     14.3,    7.9,    4.7,    0.7 ]
            fsuAnglesStanding           =   [20.8,     14.9,   11.1,    7.3,    3.3 ]
            # ivd height (dabbs method)
            ivdhLumbarDataSupine        =   [10.9,     10.1,    8.7,    8.3,    6.9 ]
            ivdhLumbarDataStanding      =   [10.55,     9.3,    8.2,   8.15,    7.0 ]


            ivdhLumbarData, fsuAngles = [],[]

            if (mode==0):
                fsuAngles = fsuAnglesStanding
                ivdhLumbarData = ivdhLumbarDataStanding
                inputValues = ["Standing", llAngleWidget.value, ivdhLumbarMinWidget.value, ivdhLumbarMaxWidget.value]

            elif (mode==1):
                fsuAngles = fsuAnglesSupine
                ivdhLumbarData = ivdhLumbarDataSupine
                inputValues = ["Supine", llAngleWidget.value, ivdhLumbarMinWidget.value, ivdhLumbarMaxWidget.value]


            # angle percentages
            fsuAnglesPct = np.divide(fsuAngles, np.sum(fsuAngles))

            # ivd height percentages
            ivdhLumbarPct = np.divide(np.subtract(ivdhLumbarData,np.min(ivdhLumbarData)),np.abs(np.min(ivdhLumbarData)-np.max(ivdhLumbarData)))
            ivdhLumbarHeights = np.add(ivdhLumbarMinWidget.value, np.multiply(ivdhLumbarPct, np.abs(ivdhLumbarMinWidget.value-ivdhLumbarMaxWidget.value)))
            ivdHeights = ivdhLumbarHeights

            llAngles = np.around(np.multiply(llAngleWidget.value, fsuAnglesPct), 2)
            
            for val in range(0, self.indivParamMatrix.columnCount):
                self.indivParamMatrix.cellWidget(0, val).text = np.flip(llAngles)[val]
                self.indivParamMatrix.cellWidget(1, val).text = np.around(np.flip(ivdHeights)[val],2)

            vtAngles = np.concatenate((vtAngles, llAngles))

        elif(edit==1): # FSU parameters changed -> calculate spine parameters
            llAngle, ivdhLMin, ivdhLMax = 0, 100, 0
            llAngles = []
            ivdHeights = []

            for i in range(0, self.indivParamMatrix.columnCount):
                val = float(self.indivParamMatrix.cellWidget(0, i).text)
                llAngle += val
                llAngles.append(val)

            for i in range(0, self.indivParamMatrix.columnCount):
                val = float(self.indivParamMatrix.cellWidget(1, i).text)
                ivdHeights.append(val)
                ivdhLMin = min(val, ivdhLMin)
                ivdhLMax = max(val, ivdhLMax)
            
            ivdHeights = np.flip(ivdHeights)
            vtAngles = np.concatenate((vtAngles, np.flip(llAngles)))


            parameterValues = [llAngle, ivdhLMin, ivdhLMax]
            for p in range (0, len(parameterWidgets)):
                parameterWidgets[p].valueChanged.disconnect(self.onEditParam)
                parameterWidgets[p].value = parameterValues[p]
                parameterWidgets[p].valueChanged.connect(self.onEditParam)
            
            if (mode==0):
                inputValues = ["Standing", llAngleWidget.value, ivdhLumbarMinWidget.value, ivdhLumbarMaxWidget.value]

            elif (mode==1):
                inputValues = ["Supine", llAngleWidget.value, ivdhLumbarMinWidget.value, ivdhLumbarMaxWidget.value]


        return [vtAngles, ivdHeights, inputValues]
    
    

    def onCreateSpineButton(self):
        logic = LumbarSpineCreatorLogic()
        logic.run(self.sawboneDirectory,
                  self.outputDirectory,
                  self.params,
                  self.outputMatrix)


#
# LumbarSpineCreatorLogic
#

class LumbarSpineCreatorLogic(ScriptedLoadableModuleLogic):


    def calcAngles(self, vectors):

        angles = []

        # calculate vertebra angles
        for index in range(0,len(vectors)-1):
            v1 = np.array(vectors[index])
            v2 = np.array(vectors[index+1])
            v1[0] = 0.0
            v2[0] = 0.0
            crossProduct = np.cross(v1, v2)
            dotProduct = np.dot(crossProduct, [-1.0, 0.0, 0.0])
            angleRad = vtk.vtkMath.AngleBetweenVectors(v1, v2)
            if (dotProduct >= 0): angleRad = np.negative(angleRad)
            angleDeg = np.round(vtk.vtkMath.DegreesFromRadians(angleRad),2)
            angles.append(angleDeg)

        return angles


    def loadDirectory(self, dirpath):
        for file in os.listdir(dirpath):
            filepath = os.path.join(dirpath, file)
            if file.endswith(".stl"):
                slicer.util.loadModel(filepath)
            if file.endswith(".json"):
                slicer.util.loadMarkups(filepath)
            if file.endswith(".h5"):
                slicer.util.loadTransform(filepath)

    def printOutput(self, inputValues, lumbarlordosisAngle, spineHeight, outputVtBodyAngles, outputFsuAngles, outputIvAngles, outputIvdhMiddle, outputIvdhAnterior, outputIvdhPosterior, outputIvdhAvrgAP, sagittalRotations, outputTableWidget, outputDirectory):
        
        for index in range(0,5):
            outputTableWidget.setCellWidget(0, index, qt.QLineEdit(str(outputFsuAngles[index])))
            outputTableWidget.setCellWidget(1, index, qt.QLineEdit(str(outputIvAngles[index])))
            outputTableWidget.setCellWidget(2, index, qt.QLineEdit(str(np.around(outputIvdhMiddle[index],2))))
            outputTableWidget.setCellWidget(3, index, qt.QLineEdit(str(np.around(outputIvdhAnterior[index],2))))
            outputTableWidget.setCellWidget(4, index, qt.QLineEdit(str(np.around(outputIvdhPosterior[index],2))))
            outputTableWidget.setCellWidget(5, index, qt.QLineEdit(str(np.around(outputIvdhAvrgAP[index],2))))
        outputTableWidget.setCellWidget(6, 0, qt.QLineEdit(str(np.around(lumbarlordosisAngle,2))))


    def transformVertebraObjects(self, transformMatrix, vtObjects):
            transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
            transformNode.SetMatrixTransformToParent(transformMatrix)
            for vtObject in vtObjects:
                if (vtObject is not None):
                    vtObject.SetAndObserveTransformNodeID(transformNode.GetID())
                    vtObject.HardenTransform()
            slicer.mrmlScene.RemoveNode(transformNode)


    def updateVtMarkers(self, vtFiducialNode):
        f = slicer.util.arrayFromMarkupsControlPoints(vtFiducialNode)

        vtPosition = np.mean([f[0], f[1], f[4], f[5]], axis=0)

        directionSlope = np.subtract(np.mean([f[0], f[1]], axis=0), np.mean([f[4], f[5]], axis=0))
        directionSlopeHeight = np.linalg.norm(directionSlope)
        directionSlope = directionSlope/directionSlopeHeight

        vtHeight = np.divide(directionSlopeHeight, 2)
        vtDirection = directionSlope

        endplateSlope = np.array(np.subtract(f[0], f[1]))
        endplateSlope[0] = 0.0
        crossProduct = np.cross([0.0, 1.0, 0.0], endplateSlope)
        dotProduct = np.dot(crossProduct, [-1.0, 0.0, 0.0])
        angleRad = vtk.vtkMath.AngleBetweenVectors([0.0, 1.0, 0.0], endplateSlope)
        if (dotProduct >= 0): angleRad = np.negative(angleRad)
        vtSlopeAngle = vtk.vtkMath.DegreesFromRadians(angleRad)

        return [vtPosition, vtHeight, vtDirection, vtSlopeAngle]
    

    # def run(self, directory, clAngle, tkAngle, llAngle, cAngle, ivdHeight):
    def run(self, directory, outputDirectory, params, outputTableWidget):

        vertebraIDs = ["S1", "L5", "L4", "L3", "L2", "L1"]
        vtRotationAngles = params[0]
        ivdHeights = params[1]
        inputValues = params[2]
        rotationAngle = vtRotationAngles[0]

        # joint Space widths  S1-L5,L5-L4,L4-L3,L3-L2,L2-L1
        jointSpacesCentral  = [1.84, 2.23, 2.14, 2.05, 1.75]
        jointSpacesInferior = [1.54, 1.77, 1.85, 1.88, 1.54]
        jointSpacesLateral  = [1.56, 1.91, 1.78, 1.78, 1.56]
        jointSpacesMedial   = [1.51, 1.73, 1.65, 1.55, 1.39]
        jointSpacesSuperior = [1.70, 2.10, 1.97, 1.92, 1.60]
        jointSpaces = [jointSpacesCentral, jointSpacesInferior, jointSpacesLateral, jointSpacesMedial, jointSpacesSuperior]
        jointSpaces = np.concatenate((jointSpaces, jointSpaces))

        # load models and fiducials in directory
        self.loadDirectory(directory)

        # get nodes and attributes
        vtModels,vtFiducials,outputVtBodyAngles,vtAngleVectors,cobbAngleVectors,ivAngleVectors,outputIvdhMiddle,outputIvdhAnterior,outputIvdhPosterior, outputIvdhAvrgAP,sagittalRotations,vtTransformJointsList,vtTransformFromPointsList = [],[],[],[],[],[],[],[],[],[],[],[],[]
        vtPositions, vtHeights, vtDirections, vtSlopeAngles = np.zeros((6,3)), np.zeros((6,1)), np.zeros((6,3)), np.zeros((6,1))

        for vt in range(0,len(vertebraIDs)):
            vtModel = slicer.mrmlScene.GetFirstNodeByName(str(vertebraIDs[vt]) + "_Sawbone")
            vtModel.GetDisplayNode().SetColor(241/255,214/255,145/255)
            vtFiducial = slicer.mrmlScene.GetFirstNodeByName(vertebraIDs[vt] + "_Fiducials")
            vtFiducial.GetDisplayNode().SetOpacity(0.0)
            vtFiducial.GetDisplayNode().SetTextScale(0.0)
            vtTransformFromPoints = slicer.mrmlScene.GetFirstNodeByName(vertebraIDs[vt] + "_TransformFromPoints")
            vtTransformJoints = slicer.mrmlScene.GetFirstNodeByName(vertebraIDs[vt] + "_Transform_Joints")
            vtModels.append(vtModel)
            vtFiducials.append(vtFiducial)
            vtTransformJointsList.append(vtTransformJoints)
            vtTransformFromPointsList.append(vtTransformFromPoints)

            # update markers
            [vtPositions[vt], vtHeights[vt], vtDirections[vt], vtSlopeAngles[vt]] = self.updateVtMarkers(vtFiducials[vt])
            

        # transform vertebra models
        for vt in range(0,len(vertebraIDs)):

            # nodes
            vtModel = vtModels[vt]
            vtTransformFromPoints = vtTransformFromPointsList[vt]
            vtTransformJoints = vtTransformJointsList[vt]
            vtObjects = [vtModel, vtFiducials[vt], vtTransformFromPoints, vtTransformJoints]

            transform = vtk.vtkTransform()
            transform.RotateX(np.negative(vtSlopeAngles[vt]))
            transform.Translate(np.negative(vtPositions[vt]))
            transformMatrix = transform.GetMatrix()
            self.transformVertebraObjects(transformMatrix, vtObjects)

            # update markers
            [vtPositions[vt], vtHeights[vt], vtDirections[vt], vtSlopeAngles[vt]] = self.updateVtMarkers(vtFiducials[vt])

        
        # change vt Angle with endplate slopes to calculate correct rotations           
            if (vt != 0):
                rotationAngle = vtSlopeAngles[vt-1] + vtRotationAngles[vt]

            transform = vtk.vtkTransform()
            transform.RotateX(rotationAngle)
            transformMatrix = transform.GetMatrix()
            self.transformVertebraObjects(transformMatrix, vtObjects)

            # update markers
            [vtPositions[vt], vtHeights[vt], vtDirections[vt], vtSlopeAngles[vt]] = self.updateVtMarkers(vtFiducials[vt])

            # calculate position for translation
            if (vt != 0):
                dir = np.add(vtDirections[vt-1],vtDirections[vt])
                ivdDir = dir/np.linalg.norm(dir)
                vtPosition = vtPositions[vt-1] + (np.multiply(vtDirections[vt-1],vtHeights[vt-1])) + np.multiply(ivdHeights[vt-1],ivdDir) + np.multiply(vtDirections[vt],vtHeights[vt])
                vtPositions[vt] = vtPosition

            # translate vertebra model
            transform = vtk.vtkTransform()
            transform.Translate(vtPositions[vt])
            transformMatrix = transform.GetMatrix()
            self.transformVertebraObjects(transformMatrix, vtObjects)



            #
            # Warping transformation to align articular processes
            #
            frw = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFiducialRegistrationWizardNode')
            frw.SetRegistrationModeToWarping()


            if(vt!=0):

                # get Nodes
                vtTransformFromPoints = vtTransformFromPointsList[vt]
                transformJointFids = vtTransformJointsList[vt-1]

                positions = slicer.util.arrayFromMarkupsControlPoints(transformJointFids)[0:40]
                jointConnectionVector = np.subtract(positions[20], positions[0])                         # vector from JointcenterL to JointcenterR
                jointConnectionVector = jointConnectionVector/np.linalg.norm(jointConnectionVector)
                jointTransformPositions = []

                #p = 0
                jointArea = 0
                for p in range(0,40,4):
                    point = positions[p]
                    normal = np.cross(np.subtract(positions[p+1],positions[p+2]), np.subtract(positions[p+1],positions[p+3]))
                    normal = normal/np.linalg.norm(normal)
                    if (np.linalg.norm(np.subtract(normal,jointConnectionVector)) > np.linalg.norm(np.add(normal, jointConnectionVector))):     # check normal direction
                        normal = np.negative(normal)
                    if (p >= 20):
                        normal = np.negative(normal)
                    jointTransformPositions.append(np.add(point, np.multiply(jointSpaces[jointArea][vt-1], normal)))
                    jointArea += 1


                # clone fiducial node to To_Points pointlist
                shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                itemIDToClone = shNode.GetItemByDataNode(vtTransformFromPoints)
                clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
                vtTransformToPoints = shNode.GetItemDataNode(clonedItemID)

                # set new joint positions to To_Points pointlist
                for t in range(0, len(jointTransformPositions)):
                    vtTransformToPoints.SetNthFiducialPositionFromArray(t, jointTransformPositions[t])


                # transform joint
                transformName = "TransformJoint_" + str(vertebraIDs[vt])
                jointTransformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', transformName)
                frw.SetOutputTransformNodeId(jointTransformNode.GetID())
                frw.SetAndObserveFromFiducialListNodeId(vtTransformFromPoints.GetID())
                frw.SetAndObserveToFiducialListNodeId(vtTransformToPoints.GetID())

                vtModel.SetAndObserveTransformNodeID(jointTransformNode.GetID())
                vtModel.HardenTransform()

            
                slicer.mrmlScene.RemoveNode(jointTransformNode)
                slicer.mrmlScene.RemoveNode(transformJointFids)
                slicer.mrmlScene.RemoveNode(vtTransformFromPoints)
                slicer.mrmlScene.RemoveNode(vtTransformToPoints)


        # calculate Output

            # find fiducials and calculate vector for angle calculation
            f = slicer.util.arrayFromMarkupsControlPoints(vtFiducials[vt])
            fi = slicer.util.arrayFromMarkupsControlPoints(vtFiducials[vt-1])

            if (vt != 0):
                outputIvdhMiddle.append(np.round(distance.euclidean(f[3], fi[2]),2))
                outputIvdhAnterior.append(distance.euclidean(f[4], fi[0]))
                outputIvdhPosterior.append(distance.euclidean(f[5], fi[1]))
                outputIvdhAvrgAP.append(distance.euclidean(np.mean([fi[0], fi[1]], axis=0), np.mean([f[4], f[5]], axis=0)))
                ivAngleVectors.append(np.subtract(f[5], f[4]))
                vtAngleVectors.append(np.subtract(f[1], f[0]))
                vtAngleVectors.append(np.subtract(f[5], f[4]))

            ivAngleVectors.append(np.subtract(f[1],f[0]))
            cobbAngleVectors.append(np.subtract(f[0],f[1]))
            sagittalRotations.append(self.calcAngles([(0.0,0.0,1.0), vtDirections[vt]])[0])

        outputVtBodyAngles = self.calcAngles(np.array(vtAngleVectors))
        outputVtBodyAngles = np.around(np.flip(outputVtBodyAngles[::2]),2)
        outputFsuAngles = np.around(np.flip(self.calcAngles(np.array(cobbAngleVectors))),2)
        outputIvAngles = self.calcAngles(np.array(ivAngleVectors))
        outputIvAngles = np.around(np.flip(outputIvAngles[::2]),2)
        outputIvdhMiddle = np.around(np.flip(outputIvdhMiddle),2)
        outputIvdhAnterior = np.around(np.flip(outputIvdhAnterior),2)
        outputIvdhPosterior = np.around(np.flip(outputIvdhPosterior),2)
        outputIvdhAvrgAP = np.around(np.flip(outputIvdhAvrgAP),2)
        sagittalRotations = np.around(np.flip(sagittalRotations),2)
        lumbarlordosisAngle = self.calcAngles([cobbAngleVectors[0],cobbAngleVectors[-1]])[0]

        pos0, pos1 = np.zeros(3), np.zeros(3)
        vtFiducials[0].GetNthFiducialPosition(0, pos0)
        vtFiducials[5].GetNthFiducialPosition(0, pos1)
        spineHeight = np.round(distance.euclidean(pos0, pos1),2)

        self.printOutput(inputValues, lumbarlordosisAngle, spineHeight, outputVtBodyAngles, outputFsuAngles, outputIvAngles, outputIvdhMiddle, outputIvdhAnterior, outputIvdhPosterior, outputIvdhAvrgAP, sagittalRotations, outputTableWidget, outputDirectory)

        print("Lumbar Spine Model was created.")




class LumbarSpineCreatorTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_LumbarSpineCreator1()

    def test_LumbarSpineCreator1(self):

        self.delayDisplay("Starting the test")

        self.delayDisplay('Test passed!')