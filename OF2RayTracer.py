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
import os
import struct

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
                if len(TempList) > 1:
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
                if len(TempList) > 1:
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
    numberOfMaterials = currIdx
    
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
        OutputList.append(fn[:-3] + "tga")
    
    # Now that that's done, add a zero to separate:
    OutputList.append(0)
    
    # Then set up the materials cache:
    for materialPath in materialDB:
        for curIdx, textureIdx in zip(materialDB[materialPath], textureDB[materialPath]):
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
            transCoords = np.dot(theseCoordinates, transMat)
            
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
            for idx in range(0, coordSubset.shape[0], 3):
                # 3 points in a triangle
                for offset in range(3):
                    v1, v2, v3 = coordSubset[idx + offset, :]
                    # Coordinate of point first:
                    OutputList.extend([v1, v2, v3])
                    # Then UV of this point:
                    u1, u2 = UVSubset[idx + offset, :]
                    OutputList.extend([u1, u2])
                # Now precompute values for Barycentric coordinate system
                A = coordSubset[idx, :]
                B = coordSubset[idx + 1, :]
                C = coordSubset[idx + 2, :]
                c = B - A
                b = C - A
                m_N = np.cross(b, c)
                
                # Now determine which is the dominant axis
                k = np.abs(m_N).argmax()
                u = (k + 1) % 3
                v = (k + 2) % 3
                
                if m_N[k] == 0.0:
                    krec = 1.0
                else:
                    krec = 1.0 / m_N[k]
                nu = m_N[u] * krec
                nv = m_N[v] * krec
                nd = np.dot(m_N, A) * krec
                # First line of equation
                if (b[u] * c[v] - b[v] * c[u]) == 0.0:
                    reci = 1.0
                else:
                    reci = 1.0 / (b[u] * c[v] - b[v] * c[u])
                bnu = b[u] * reci
                bnv = -b[v] * reci
                # Second line of equation
                cnu = c[v] * reci
                cnv = -c[u] * reci
                if (m_N ** 2).sum() == 0.0:
                    m_N_norm = m_N
                else:
                    m_N_norm = m_N / np.sqrt((m_N ** 2).sum())
                
                # Now enter these variables
                OutputList.append(k)
                OutputList.extend(c.tolist())
                OutputList.extend(b.tolist())
                OutputList.extend(m_N.tolist())
                OutputList.extend(m_N_norm.tolist())
                # And then the remaining floats
                OutputList.extend([nu, nv, nd, bnu, bnv, cnu, cnv])
            
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
    
    outFile = open(filename, 'wb')
    # Number of materials
    matCount = int(listin.pop(0))
    outFile.write(struct.pack('i', matCount))
    # Number of textures
    textCount = int(listin.pop(0))
    outFile.write(struct.pack('i', textCount))
    
    # And now the list of filenames:
    for idx in range(textCount):
        strSize = listin.pop(0)
        outFile.write(struct.pack('i', strSize))
        outFile.write(struct.pack(str(strSize) + 's', listin.pop(0)))
    zeroCheck = listin.pop(0)
    if zeroCheck != 0:
        outFile.close()
        raise Exception("Error encountered entering filenames. Failed zero check.")
    outFile.write(struct.pack('i', 0))
    
    # Now to process the material indices and texture indices:
    # Let's do this as a batch:
    subList = listin[:(2*matCount)]
    listin = listin[(2*matCount):]
    outFile.write(struct.pack('%si' % (2 * matCount), *subList))
    # for idx in range(matCount):
    #     # Material Index
    #     outFile.write(struct.pack('i', listin.pop(0)))
    #     # Texture Index
    #     outFile.write(struct.pack('i', listin.pop(0)))
    zeroCheck = listin.pop(0)
    if zeroCheck != 0:
        outFile.close()
        raise Exception("Error encountered pairing materials with textures. Failed zero check.")
    
    outFile.write(struct.pack('i', 0))
    
    verticesList = []
    grpCount = 0
    
    # Finally, go through triangles and create objects. Do this until the list is empty:
    while len(listin) > 0:
        # Expect a number of triangles:
        tempList = []
        triCount = int(listin.pop(0))
        tempList.append(triCount)
        grpCount += 1
        print "Processing triangle group %i..." %grpCount
        for idx in range(triCount):
            # Each triangle
            # Let's do this as a batch
            subList = np.array(listin[:15])
            listin = listin[15:]
            if np.any(subList >= 65536):
                outFile.close()
                raise Exception("Point axis or UV value overflow in fixed point conversion")
            # Now convert to a list again and then write the contents to the file:
            subList = (subList * 65536).astype(int).tolist()
            tempList.extend(subList)
            # for PointIdx in range(3):
            #     # Corners of triangle
            #     for AxisIdx in range(3):
            #         # for u, v and w
            #         PointVal = listin.pop(0)
            #         if PointVal > 65536:
            #             raise Exception("Point axis value overflow in fixed point conversion")
            #         outFile.write(struct.pack('i', int(PointVal * 65536)))
            #     # UV coordinates
            #     for AxisIdx in range(2):
            #         PointVal = listin.pop(0)
            #         if PointVal > 65536:
            #             raise Exception("Point UV value overflow in fixed point conversion")
            #         outFile.write(struct.pack('i', int(PointVal * 65536)))
            # Now to enter precomputed values:
            # k:
            tempList.append(int(listin.pop(0)))
            # c, b, m_N, m_N_norm:
            # 4 vectors + 7 values = 12 + 7 = 19
            
            subList = np.array(listin[:19])
            listin = listin[19:]
            if np.any(subList >= 65536):
                outFile.close()
                raise Exception("Pre-calculation value overflow in fixed point conversion")
            # Now convert the list again and then write the contents to a file:
            subList = (subList * 65536).astype(int).tolist()
            tempList.extend(subList)
            
            # for PointIdx in range(4):
            #     # For the different variables
            #     for Axis in range(3):
            #         PointVal = listin.pop(0)
            #         if PointVal > 65536:
            #             raise Exception("Pre-calculation value overflow in fixed point conversion")
            #         outFile.write(struct.pack('i', int(PointVal * 65536)))
            
            # Seven remaining items:
            # for item in range(7):
            #     outFile.write(struct.pack('i', int(listin.pop(0) * 65536)))
        # Now to get the material index:
        tempList.append(listin.pop(0))
        # Then zero check:
        zeroCheck = listin.pop(0)
        if zeroCheck != 0:
            outFile.close()
            raise Exception("Error encountered pairing triangle points with UV values. Failed zero check.")
        tempList.append(0)
        verticesList.extend(tempList)
    print "Triangle processing complete. Now attempting to write data to file..."
    outFile.write(struct.pack('%ii' % len(verticesList), *verticesList))
    print "Done."
    outFile.close()

def AIO(dictIn, filename = "World.crt"):
    """
    AIO extracts data from a dictionary and writes the output directly to file.
    
    This function avoids having to use two functions (one to read and the other
    to write) which can add increased overhead.
    
    This function doesn't return anything.
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
                if len(TempList) > 1:
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
                if len(TempList) > 1:
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
    numberOfMaterials = currIdx
    
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
        OutputList.append(fn[:-3] + "tga")
    
    # Now that that's done, add a zero to separate:
    OutputList.append(0)
    
    # Then set up the materials cache:
    for materialPath in materialDB:
        for curIdx, textureIdx in zip(materialDB[materialPath], textureDB[materialPath]):
            OutputList.append(int(curIdx))
            OutputList.append(int(textureIdx))
    
    # Again, add a zero to separate:
    OutputList.append(0)
    
    #####################################################################################
    # At this point, write everything to file. We can then clear the stack and then write
    # the triangle data to the empty stack meaning we can write data the to file directly.
    
    outFile = open(filename, 'wb')
    # Number of materials
    matCount = int(OutputList.pop(0))
    outFile.write(struct.pack('i', matCount))
    # Number of textures
    textCount = int(OutputList.pop(0))
    outFile.write(struct.pack('i', textCount))
    
    # And now the list of filenames:
    for idx in range(textCount):
        strSize = OutputList.pop(0)
        outFile.write(struct.pack('i', strSize))
        outFile.write(struct.pack(str(strSize) + 's', OutputList.pop(0)))
    zeroCheck = OutputList.pop(0)
    if zeroCheck != 0:
        outFile.close()
        raise Exception("Error encountered entering filenames. Failed zero check.")
    outFile.write(struct.pack('i', 0))
    
    # Now to process the material indices and texture indices:
    # Let's do this as a batch:
    subList = OutputList[:(2*matCount)]
    OutputList = OutputList[(2*matCount):]
    outFile.write(struct.pack('%si' % (2 * matCount), *subList))
    # for idx in range(matCount):
    #     # Material Index
    #     outFile.write(struct.pack('i', listin.pop(0)))
    #     # Texture Index
    #     outFile.write(struct.pack('i', listin.pop(0)))
    zeroCheck = OutputList.pop(0)
    if zeroCheck != 0:
        outFile.close()
        raise Exception("Error encountered pairing materials with textures. Failed zero check.")
    
    outFile.write(struct.pack('i', 0))
    
    
    if len(OutputList) > 0:
        print "WARNING: There are still items within the stack!"
    
    #####################################################################################
    # Clear the list ready for the next lot of work
    OutputList = []
    
    
    # This is when things get complicated. Go through the list of transformations:
    for transIdx, niceFilename in enumerate(TransformationList):
        print "Processing triangle group %i of %i..." % (transIdx + 1, len(TransformationList))
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
                transMat = np.vstack((transMat[:3, :3], transMat[3, :3])) # np.vstack((transMat[:3, :3].T, transMat[3, :3]))
            else:
                # There isn't a transformation matrix. Use identity matrix
                transMat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
            # Before finally multiplying through
            transCoords = np.dot(theseCoordinates, transMat)
            
            # Next, extract and append the grouped components:
            groupedCoords = np.vstack([groupedCoords, transCoords])
            groupedPatterns = np.vstack([groupedPatterns, thesePatterns])
            groupedUVCoords = np.vstack([groupedUVCoords, theseUVCoords])
        
        # Now we have all the necessary inforamtion, it's time to start going through pattern
        # indices and appending to the master OutputList variable
        for textIdx, matIdx in enumerate(materialDB[niceFilename]):
            print "\tProcessing texture %i and material %i" % (textIdx, matIdx)
            # Obtain the subset of coordinates based on the pattern index:
            coordSubset = groupedCoords[groupedPatterns.flatten() == textIdx, :]
            UVSubset = groupedUVCoords[groupedPatterns.flatten() == textIdx, :]
            
            # Now compute how many triangles we're dealing with and append this number to the list:
            noTrianglesSubset = coordSubset.shape[0] / 3
            OutputList.append(noTrianglesSubset)
            
            # Then go through each triangle componenet and add this to the list:
            for idx in range(0, coordSubset.shape[0], 3):
                # 3 points in a triangle
                for offset in range(3):
                    if np.any(coordSubset[idx + offset, :] > 32767.) or np.any(coordSubset[idx + offset, :] < -32768.):
                        print "\t\tCorner %i point coordinates of triangle %i exceeded threshold %s" % (offset, (idx / 3) + 1, str(coordSubset[idx + offset, :]))
                    v1, v2, v3 = 65536 * coordSubset[idx + offset, :]
                    # Coordinate of point first:
                    OutputList.extend([int(v1), int(v2), int(v3)])
                    # Then UV of this point:
                    u1, u2 = 65536 * UVSubset[idx + offset, :]
                    if np.any(UVSubset[idx + offset, :] > 32767.) or np.any(UVSubset[idx + offset, :] < -32768.):
                        print "\t\tCorner %i UV coordinates of triangle %i exceeded threshold %s" % (offset, (idx / 3) + 1, str(UVSubset[idx + offset, :]))
                    OutputList.extend([int(u1), int(u2)])
                # Now precompute values for Barycentric coordinate system
                A = coordSubset[idx, :]
                B = coordSubset[idx + 1, :]
                C = coordSubset[idx + 2, :]
                c = B - A
                b = C - A
                m_N = np.cross(b, c)
                
                # Now determine which is the dominant axis
                k = np.abs(m_N).argmax()
                u = (k + 1) % 3
                v = (k + 2) % 3
                
                if m_N[k] == 0.0:
                    krec = 1.0
                else:
                    krec = 1.0 / m_N[k]
                nu = m_N[u] * krec
                nv = m_N[v] * krec
                nd = np.dot(m_N, A) * krec
                # First line of equation
                if (b[u] * c[v] - b[v] * c[u]) == 0.0:
                    reci = 1.0
                else:
                    reci = 1.0 / (b[u] * c[v] - b[v] * c[u])
                bnu = b[u] * reci
                bnv = -b[v] * reci
                # Second line of equation
                cnu = c[v] * reci
                cnv = -c[u] * reci
                if (m_N ** 2).sum() == 0.0:
                    m_N_norm = m_N
                else:
                    m_N_norm = m_N / np.sqrt((m_N ** 2).sum())
                
                # Now enter these variables
                if np.any(c > 32767.) or np.any(c < -32768.):
                    print "\t\tVariable c on triangle %i exceeded threshold: %s" % ( (idx / 3) + 1, str(c))
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 3
                    c = np.zeros(c.shape)
                if np.any(b > 32767.) or np.any(b < -32768.):
                    print "\t\tVariable b on triangle %i exceeded threshold: %s" % ( (idx / 3) + 1, str(b))
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 4
                    b = np.zeros(b.shape)
                if np.any(m_N > 32767.) or np.any(m_N < -32768.):
                    print "\t\tVariable m_N on triangle %i exceeded threshold: %s" % ( (idx / 3) + 1, str(m_N))
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 5
                    m_N = np.zeros(m_N.shape)
                if np.any(m_N_norm > 32767.) or np.any(m_N_norm < -32768.):
                    print "\t\tVariable m_N_norm on triangle %i exceeded threshold: %s" % ( (idx / 3) + 1, str(m_N_norm))
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 6
                    m_N_norm = np.zeros(m_N_norm.shape)
                if nu > 32767. or nu < -32768.:
                    print "\t\tVariable nu on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, nu)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 7
                    nu = 0
                if nv > 32767. or nv < -32768.:
                    print "\t\tVariable nv on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, nv)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 8
                    nv = 0
                if nd > 32767. or nd < -32768.:
                    print "\t\tVariable nd on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, nd)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 9
                    nd = 0
                if bnu > 32767. or bnu < -32768.:
                    print "\t\tVariable bnu on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, bnu)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 10
                    bnu = 0
                if bnv > 32767. or bnv < -32768.:
                    print "\t\tVariable bnv on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, bnv)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t cnu: %s" % str(cnu)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 11
                    bnv = 0
                if cnu > 32767. or cnu < -32768.:
                    print "\t\tVariable cnu on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, cnu)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnv: %s" % str(cnv)
                    k = 12
                    cnu = 0
                if cnv > 32767. or cnv < -32768.:
                    print "\t\tVariable cnv on triangle %i exceeded threshold: %f" % ( (idx / 3) + 1, cnv)
                    print "\t\t\t A: %s" % str(A)
                    print "\t\t\t B: %s" % str(B)
                    print "\t\t\t C: %s" % str(C)
                    print "\t\t\t k: %s" % str(k)
                    print "\t\t\t reci: %s" % str(reci)
                    print "\t\t\t c: %s" % str(c)
                    print "\t\t\t b: %s" % str(b)
                    print "\t\t\t m_N: %s" % str(m_N)
                    print "\t\t\t m_N_norm: %s" % str(m_N_norm)
                    print "\t\t\t nu: %s" % str(nu)
                    print "\t\t\t nv: %s" % str(nv)
                    print "\t\t\t nd: %s" % str(nd)
                    print "\t\t\t bnu: %s" % str(bnu)
                    print "\t\t\t bnv: %s" % str(bnv)
                    print "\t\t\t cnu: %s" % str(cnu)
                    k = 13
                    cnv = 0
                OutputList.append(k)
                OutputList.extend((65536 * c).astype(int).tolist())
                OutputList.extend((65536 * b).astype(int).tolist())
                OutputList.extend((65536 * m_N).astype(int).tolist())
                OutputList.extend((65536 * m_N_norm).astype(int).tolist())
                # And then the remaining floats
                OutputList.extend([int(65536 * nu), int(65536 * nv), int(65536 * nd), int(65536 * bnu), int(65536 * bnv), int(65536 * cnu), int(65536 * cnv)])
            
            # This concludes all the triangles. Now send the material index:
            OutputList.append(matIdx)
            # Then send a zero to denote the end of the record:
            OutputList.append(0)
    
    # Finally, save the list to the file
    outFile.write(struct.pack("%si" % str(len(OutputList)), *OutputList))
    outFile.close()
    print "Done."