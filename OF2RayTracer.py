"""
This file automates the process of world extraction from OpenFlight files.

This supersedes the Extraction.py file. Functionality that's been brought
across with this includes the extraction of models (including models that
reference other models, etc.) and the writing of byte code that is 
interpreted by the C Ray tracer.

 Author: Andrew Hills (a.hills@sheffield.ac.uk)
Version: 0.0.1 (01/04/2014)
"""

import numpy as np

def ExtractWorld(dictIn):
    """
    ExtractWorld extracts world information from an OpenFlight file. This
    function requires a dictionary object that is produced by the OpenFlight
    parser.
    
    This function returns a list which can then be written to by the byte code
    writer function.
    """
    # Define some functions for expanding the objects
    def ExpandSearch(listObject):
        FocussedRecordsList = []
        TempList = []
        for item in listObject:
            if isinstance(item, list):
                FocussedRecordsList.extend(ExpandSearch(item))
            # Not a list, so we must have a record.
            elif isinstance(item, dict):
                # Yes, this is a record.
                if 'Datatype' not in item:
                    continue
                if item['Datatype'] != "ExternalReference":
                    continue
                # This record is appropriate
                if len(TempList) > 2:
                    FocussedRecordsList.append(tuple(TempList))
                    TempList = []
                if len(TempList) == 1:
                    # An item is already on the stack. Check to see if it was a dict
                    if isinstance(TempList[0], dict):
                        # We should append without a matrix record. Insert a None instead
                        TempList.append(None)
                        FocussedRecordsList.append(tuple(TempList))
                        TempList = []
                        continue
                    elif isinstance(TempList[0], np.ndarray):
                        # Things look to be reversed. Insert this before the matrix
                        TempList.insert(0, item)
                        continue
                # If here, we should append this item
                TempList.append(item)
            elif isinstance(item, np.ndarray):
                # This is a matrix record
                if len(TempList) > 2:
                    FocussedRecordsList.append(tuple(TempList))
                    TempList = []
                if len(TempList) == 1:
                    # An item is already on the stack. Check to see if it was a dict
                    if isinstance(TempList[0], np.ndarray):
                        # A matrix followed by a matrix is unusual. Remove the previous one
                        TempList = [item]
                        continue
                # If here, it's safe to append the item
                TempList.append(item)
        return FocussedRecordsList
    
    # The following function is essentially a copy of the previous VertexListToComplexTextureCoords
    # function but this one doesn't rename the output.
    def GetTextureCoords(dictIn):
        """
            The input to this function should be the complete records dictionary.
        
            This function outputs a dictionary of NumPy coordinates.
        """
        if not isinstance(dictIn, dict):
            raise Exception('Input it not a dictionary object.')
        
        if 'External' not in dictIn:
            raise Exception('input is not a valid dictionary object.')
        
        newObject = dict()
        
        # This is similar to the ExtractExternal function.
        for key in dictIn['External']:
            tempList = []
            tempList2 = []
            
            if 'VertexList' not in dictIn['External'][key]:
                continue
            
            for item, txpidx in zip(dictIn['External'][key]['VertexList'], dictIn['External'][key]['TexturePatterns']):
                tempMat = None
                for idx, offset in enumerate(item):
                    if tempMat is None:
                        tempMat = np.zeros((len(item), max(dictIn['External'][key]['Vertices'][offset]['TextureCoordinate'].shape)))
                    tempMat[idx, :] = dictIn['External'][key]['Vertices'][offset]['TextureCoordinate']
                tempList.append(tempMat)
                tempList2.append(np.ones((len(item), 1)) * txpidx)
            if len(tempList) > 0:
                newObject[key] = dict()
                newObject[key]['Coords'] = np.vstack(tempList)
                newObject[key]['TexturePattern'] = np.vstack(tempList2)
        return newObject
    
    # Similarly, this is a copy of the previously seen VertexListToCoords function but also
    # doesn't rename the key.
    def GetCoords(dictIn):
        """
            The input to this function should be the complete records dictionary.
        
            This function outputs a dictionary of NumPy coordinates.
        """
        if not isinstance(dictIn, dict):
            raise Exception('Input it not a dictionary object.')
            
        if 'External' not in dictIn:
            raise Exception('input is not a valid dictionary object.')
        
        newObject = dict()
        
        # This is similar to the ExtractExternal function.
        for key in dictIn['External']:
            tempList = []
            
            if 'VertexList' not in dictIn['External'][key]:
                continue
            
            for item, scale, translate in zip(dictIn['External'][key]['VertexList'], dictIn['External'][key]['Scale'], dictIn['External'][key]['Translate']):
                tempMat = None
                for idx, offset in enumerate(item):
                    if tempMat is None:
                        tempMat = np.zeros((len(item), max(dictIn['External'][key]['Vertices'][offset]['Coordinate'].shape)))
                    tempMat[idx, :] = dictIn['External'][key]['Vertices'][offset]['Coordinate']
                tempList.append(tempMat * scale + translate)
            if len(tempList) > 0:
                newObject[key] = np.vstack(tempList)
        return newObject
    
    ###########################################################
    # Now back to the main function:
    
    # Obtain a list of models and their transformation matrices
    RecordsList = ExpandSearch(dictIn['Tree'])
    
    TransformationList = dict()
    
    # Declare some primary lists
    PrimaryTextureFiles = []
    PrimaryModelFiles = []
    
    # Populate the list by expanding the records. These filenames can be used to
    # open the files in the External list
    for item in RecordsList:
        if item[0]['NewFilename'] not in TransformationList:
            TransformationList[item[0]['NewFilename']] = []
        TransformationList[item[0]['NewFilename']].append(item[1])
        if item[0]['NewFilename'] not in PrimaryModelFiles:
            PrimaryModelFiles.append(item[0]['NewFilename'])
    
    # Assume that all models were added to the PrimaryModelFiles variable. This
    # should leave behind textures. Save the remainder as textures:
    for item in dictIn['External']:
        if item not in PrimaryModelFiles:
            PrimaryTextureFiles.append(item)
    
    textureDB = dict()
    materialDB = dict()
    currIdx = 0
    # Now for the fun part. We've got to construct a dictionary for each model file
    # that maps its texture files to the appropriate primary texture. Additionally,
    # there needs to be a counter to map the texture file. Let's also go through
    # each texture and create an additional object which relates to the appropriate
    # material index.
    for modelPath in PrimaryModelFiles:
        model = dictIn['External'][modelPath]
        textureDB[modelPath] = []
        materialDB[modelPath] = range(currIdx, currIdx + len(model['Textures']))
        currIdx += len(model['Textures'])
        for textureRecord in model['Textures']:
            # Each of these records seem to be in sequential order. Thus, we can
            # assume that a list object will be sufficient in translating the 
            # texture pattern index to the appropriate texture. Thus, we simply
            # couple the texture index with the primary texture index.
            textureDB[modelPath].append(PrimaryTextureFiles.index(textureRecord['Filename']))
    # And we can then get this useful stat.
    numberOfMaterials = currIdx + 1
    
    # Now let's go through the primary texture files and correct the file separators
    # and remove the path. This requires us to go through the primary list and alter
    # wherever necessary
    PrimaryTextureFilenames = []
    for filen in PrimaryTextureFiles:
        filen.replace('\\', os.path.sep)
        if filen.count(os.path.sep) > 0:
            PrimaryTextureFilenames.append(filen[filen.rindex(os.path.sep)+1:])
        else:
            PrimaryTextureFilenames.append(filen)
    # And now we have corrected the texture filenames
    
    # So now let's get the coordinates for all the models in a nice dictionary form:
    coordinates = GetCoords(dictIn)
    textureCoordinates = GetTextureCoords(dictIn)
    
    # And this is where we part our ways from the Extraction.py script.
    ###################################################################
    ###################################################################
    # Create an output list in preparation for storing information.
    OutputList = []
    
    # The first thing to do is append the number of materials
    OutputList.append(numberOfMaterials)
    
    # It's now necessary to add texture filenames. Provide the number that's needed.
    OutputList.append(len(PrimaryTextureFilenames))
    
    for textIdx, fn in enumerate(PrimaryTextureFilenames):
        # This will make reading strings easier in C
        OutputList.append(len(fn))
        OutputList.append(fn[:-3] + ".tga")
    
    # Now that that's done, add a zero to separate:
    OutputList.append(0)
    
    # Then set up the materials cache:
    for materialPath in materialDB:
        for curIdx, texutreIdx in zip(materialDB[materialPath], textureDB[materialPath]):
            OutputList.append(int(curIdx))
            OutputList.append(int(textureIdx))
    
    # Again, add a zero to separate:
    OutputList.append(0)
    
    # This is when things get complicated. Go through the list of transformations:
    for niceFilename in TransformationList:
        theseCoordinates = np.hstack((coordinates[niceFilename], np.ones((coordinates[niceFilename].shape[0], 1))))
        theseUVCoords = textureCoordinates[niceFilename]['Coords']
        thesePatterns = textureCoordinates[niceFilename]['TexturePattern'] - np.min(textureCoordinates[niceFilename]['TexturePattern'].flatten())
        
        # Initialise grouped components:
        groupedCoords = np.zeros((0, 3))
        groupedPatterns = np.zeros((0, 1))
        groupedUVCoords = np.zeros((0, 2))
        
        # Now extract the transformation matrix:
        for transIdx, transMat in enumerate(TransformationList[niceFilename]):
            # Check what the transformation matrix variable contains:
            if transMat is not None:
                # retrieve the necessary components:
                transMat = np.vstack((transMat[:3, :3].T, transMat[3, :3]))
            else:
                # There isn't a transformation matrix. Use identity matrix
                transMat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
            # Before finally multiplying through
            transCoords = np.dot(theseCoordiantes, transMat)
            
            # Next, extract and append the grouped components:
            groupedCoords = np.vstack([groupedCoords, transCoords])
            groupedPatterns = np.vstack([groupedPatterns, thesePatterns])
            groupedUVCoords = np.vstack([groupedUVCoords, theseUVCoords])
        
        # Now we have all the necessary inforamtion, it's time to start going through pattern
        # indices and appending to the master OutputList variable
        for textIdx, matIdx in enumerate(materialDB[niceFilename]):
            # Obtain the subset of coordinates based on the pattern index:
            coordSubset = groupedCoords[groupedPatterns.flatten() == textIdx, :]
            UVSubset = groupedUVCoords[groupedPatterns.flatten() == textIdx, :]
            
            # Now compute how many triangles we're dealing with and append this number to the list:
            noTrianglesSubset = coordSubset.shape[0] / 3
            OutputList.append(noTrianglesSubset)
            
            # Then go through each triangle componenet and add this to the list:
            for idx in range(0, coordSubset, 3):
                # 3 points in a triangle
                for offset in range(3):
                    v1, v2, v3 = coordSubset[idx + offset, :]
                    # Coordinate of point first:
                    OutputList.extend([v1, v2, v3])
                    # Then UV of this point:
                    u1, u2 = UVSubset[idx + offset, :]
                    outputList.extend([u1, u2])
            # This concludes all the triangles. Now send the material index:
            OutputList.append(matIdx)
            # Then send a zero to denote the end of the record:
            OutputList.append(0)
    
    # Finally, return the list object
    return OutputList
    
def ByteCodeWriter(listin, filename="World.crt"):
    """
    ByteCodeWriter converts the input list to byte-based codes that can be read
    by the C Ray Tracer.
    
    This function creates a file (default: World.crt).
    """
    
    pass