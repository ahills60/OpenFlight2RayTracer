import numpy as np
import quickio as qio
import os

def MegaSave(dictIn, filename = "World.crt"):
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
    megaMatrix = np.zeros((0, 35))
    
    
    # This is when things get complicated. Go through the list of transformations:
    for transIdx, niceFilename in enumerate(TransformationList):
        print "Processing triangle group %i of %i..." % (transIdx, len(TransformationList))
        theseCoordinates = np.hstack((coordinates[niceFilename], np.ones((coordinates[niceFilename].shape[0], 1))))
        theseUVCoords = textureCoordinates[niceFilename]['Coords']
        thesePatterns = textureCoordinates[niceFilename]['TexturePattern'] - np.min(textureCoordinates[niceFilename]['TexturePattern'].flatten())
        
        # Initialise grouped components:
        groupedCoords = np.zeros((0, 3))
        groupedPatterns = np.zeros((0, 1))
        groupedUVCoords = np.zeros((0, 2))
        
        # Now extract the transformation matrix:
        for transMat in TransformationList[niceFilename]:
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
            print "\tProcessing texture %i and material %i" % (textIdx, matIdx)
            # Obtain the subset of coordinates based on the pattern index:
            coordSubset = groupedCoords[groupedPatterns.flatten() == textIdx, :]
            UVSubset = groupedUVCoords[groupedPatterns.flatten() == textIdx, :]
            
            # Now compute how many triangles we're dealing with and append this number to the list:
            noTrianglesSubset = coordSubset.shape[0] / 3
            
            # Then go through each triangle componenet and add this to the list:
            for idx in range(0, coordSubset.shape[0], 3):
                # Set up a temp store
                tempStore = np.zeros((1, 35))
                # 3 points in a triangle
                for offset in range(3):
                    # Coordinates
                    tempStore[0, (5 * offset):(5 * offset + 3)] = coordSubset[idx + offset, :]
                    
                    # Then UV of this point:
                    tempStore[0, (5 * offset + 3):(5 * offset + 5)] = UVSubset[idx + offset, :]
                    
                    
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
                
                tempStore[0, 15] = k
                tempStore[0, 16:19] = c
                tempStore[0, 19:22] = b
                tempStore[0, 22:25] = m_N
                tempStore[0, 25:28] = m_N_norm
                # And then the remaining floats
                tempStore[0, 28] = nu
                tempStore[0, 29] = nv
                tempStore[0, 30] = nd
                tempStore[0, 31] = bnu
                tempStore[0, 32] = bnv
                tempStore[0, 33] = cnu
                tempStore[0, 34] = cnv
                megaMatrix = np.vstack((megaMatrix, tempStore))
            
    qio.save('mega.mat', 'megaMatrix')
    print "Done."