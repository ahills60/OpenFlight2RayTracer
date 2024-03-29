# This is a script that will open a dictionary object outputted from the OpenFlight class.
# The script opens the external list of objects and processes each one, extracting the
# coordinates. The output is another dictionary object but with only coordinate information

import numpy as np
import os

def ExtractExternal(dictIn):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs coordinates in NumPy format.
        
         Author: Andrew Hills
        Version: 0.0.1
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input is not a dictionary object.')
    
    if 'External' not in dictIn:
        raise Exception('Input is not a valid dictionary object.')
    
    newObject = dict()
    
    # This should be a valid dictionary file:
    for key in dictIn['External']:
        # This will go through each key:
        print "Processing " + str(key) + "..."
        tempMat = np.zeros((0, 3))
        if 'Tree' not in dictIn['External'][key]:
            continue
        for item in dictIn['External'][key]['Tree']:
            # This will go through the list of objects
            if not isinstance(item, dict):
                continue
            if 'Coordinate' in item:
                tempMat = np.vstack((tempMat, item['Coordinate']))
        if tempMat.shape[0] > 0:
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = tempMat
    return newObject


def ExtractExternalRecords(dictIn):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs records that hold coordinate information.
        
         Author: Andrew Hills
        Version: 0.0.1
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    if 'External' not in dictIn:
        raise Exception('input is not a valid dictionary object.')
    
    newObject = dict()
    
    # This is similar to the ExtractExternal function.
    for key in dictIn['External']:
        # Go through each key:
        print "Processing", str(key) + "..."
        
        tempList = []
        if 'Tree' not in dictIn['External'][key]:
            continue
        for item in dictIn['External'][key]['Tree']:
            # This will go through the list of objects:
            if not isinstance(item, dict):
                continue
            if 'Coordinate' in item:
                tempList.append(item)
        if len(tempList) > 0:
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = tempList
    return newObject

def TellMeDataTypes(dictIn):
    """
        The input to this function should be a dictionary object that contains
        lists of records that hold the "Coordinate" parameter.
        
        This function will print the object types.
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input is not a dictionary object.')
    
    if 'External' in dictIn:
        # Then process this first and output the results
        TellMeDataTypes(dictIn)
        return
    
    for key in dictIn:
        # Print out the name and leave the line hanging.
        print "Variable", str(key) + ": ",
        SeenElements = []
        for item in dictIn[key]:
            if 'Datatype' not in item:
                continue
            if item['Datatype'] not in SeenElements:
                print item['Datatype'] + ", ",
                SeenElements.append(item['Datatype'])
        print "\n",
    print "\n"


def GetVertexLists(dictIn):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs vertex list records.
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    if 'External' not in dictIn:
        raise Exception('input is not a valid dictionary object.')
    
    newObject = dict()
    
    def ProcessList(listObject):
        vList = []
        if not isinstance(listObject, list):
            return
        # This is a list
        for item in listObject:
            if isinstance(item, list):
                tempList = ProcessList(item)
                if len(tempList) > 0:
                    vList.extend(tempList)
            elif isinstance(item, dict):
                if 'Datatype' not in item:
                    continue
                else:
                    if item['Datatype'] == 'VertexList':
                        vList.append(item)
        return vList
    
    
    # This is similar to the ExtractExternal function.
    for key in dictIn['External']:
        # Go through each key:
        print "Processing", str(key) + "..."
        
        tempList = []
        if 'Tree' not in dictIn['External'][key]:
            continue
        for item in dictIn['External'][key]['Tree']:
            # This will go through the list of objects:
            if isinstance(item, list):
                tempList.extend(ProcessList(item))
            elif isinstance(item, dict):
                if 'Datatype' not in item:
                    continue
                if item['Datatype'] == "VertexList":
                    tempList.append(item)
        if len(tempList) > 0:
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = tempList
    return newObject

def VertexListToCoords(dictIn):
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
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = np.vstack(tempList)
    return newObject
    
def VertexListToCoordsMaster(dictIn):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs a dictionary of NumPy coordinates.
    """
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    # This is similar to the ExtractExternal function.
    tempList = []
    
    if 'VertexList' not in dictIn:
        raise Exception("Unable to find the vertex list.")
    
    for item, scale, translate in zip(dictIn['VertexList'], dictIn['Scale'], dictIn['Translate']):
        tempMat = None
        for idx, offset in enumerate(item):
            if tempMat is None:
                tempMat = np.zeros((len(item), max(dictIn['Vertices'][offset]['Coordinate'].shape)))
            tempMat[idx, :] = dictIn['Vertices'][offset]['Coordinate']
        tempList.append(tempMat * scale + translate)
    if len(tempList) > 0:
        newObject = np.vstack(tempList)
    return newObject


def CreateHeaderFiles(modelName, dictIn, filename = "scene.h", scale = 20):
    coordinates = VertexListToCoords(dictIn)
    if modelName not in coordinates:
        raise Exception("Unable to find model \"" + modelName + "\"")
    
    # Simplify dictionary object
    coordinates = coordinates[modelName]
    
    HEADER = ["#ifndef SCENE_H_\n",
        "#define SCENE_H_\n",
        "\n",
        "#include \"scenecalcs.h\"\n",
        "#include \"renderscene.h\"\n",
        "\n",
        "// Called to draw scene\n",
        "void RenderScene(void)\n",
        "{\n",
        "float normal[3];	// Storage for calculated surface normal\n",
        "\n",
        "// Clear the window with current clearing colour\n",
        "glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);\n",
        "\n",
        "// Save the matrix state and do the rotations\n",
        "glPushMatrix();\n",
        "glRotatef(xRot, 1.0f, 0.0f, 0.0f);\n",
        "glRotatef(yRot, 0.0f, 1.0f, 0.0f);\n",
        "\n",
        "\n",
        "// Nose Cone /////////////////////////////\n",
        "// Set material colour\n",
        "glColor3ub(128, 128, 128);\n",
        "glBegin(GL_TRIANGLES);\n"]
    
    FOOTER = ["glEnd();\n",
        "}\n"
        "\n",
        "// Restore the matrix state\n",
        "glPopMatrix();\n",
        "// Display the results\n",
        "glutSwapBuffers();\n",
        "}\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    
    for idx in range(0, coordinates.shape[0], 3):
        if idx > 0:
            outFile.write('}\n\n')
        v1, v2, v3 = scale * coordinates[idx, :]
        outFile.write("{\n")
        outFile.write("float v[3][3] = {{%ff, %ff, %ff},\n" % (v1, v2, v3)),
        
        v1, v2, v3 = scale * coordinates[idx + 1, :]
        outFile.write("     {%ff, %ff, %ff},\n" % (v1, v2, v3))
        
        v1, v2, v3 = scale * coordinates[idx + 2, :]
        outFile.write("     {%ff, %ff, %ff}};\n" % (v1, v2, v3))
        
        outFile.write("calcNormal(v, normal);\n")
        outFile.write("glNormal3fv(normal);\n")
        outFile.write("glVertex3fv(v[0]);\n")
        outFile.write("glVertex3fv(v[1]);\n")
        outFile.write("glVertex3fv(v[2]);\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()

def VertexListToTextureCoords(dictIn):
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
        
        for item in dictIn['External'][key]['VertexList']:
            tempMat = None
            for idx, offset in enumerate(item):
                if tempMat is None:
                    tempMat = np.zeros((len(item), max(dictIn['External'][key]['Vertices'][offset]['TextureCoordinate'].shape)))
                tempMat[idx, :] = dictIn['External'][key]['Vertices'][offset]['TextureCoordinate']
            tempList.append(tempMat)
        if len(tempList) > 0:
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = np.vstack(tempList)
    return newObject

def GetThisRecord(dictIn, recordName):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs vertex list records.
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    if 'External' not in dictIn:
        raise Exception('input is not a valid dictionary object.')
    
    newObject = dict()
    
    def ProcessList(listObject):
        vList = []
        if not isinstance(listObject, list):
            return
        # This is a list
        for item in listObject:
            if isinstance(item, list):
                tempList = ProcessList(item)
                if len(tempList) > 0:
                    vList.extend(tempList)
            elif isinstance(item, dict):
                if 'Datatype' not in item:
                    continue
                else:
                    if item['Datatype'] == recordName:
                        vList.append(item)
        return vList
    
    
    # This is similar to the ExtractExternal function.
    for key in dictIn['External']:
        # Go through each key:
        print "Processing", str(key) + "..."
        
        tempList = []
        if 'Tree' not in dictIn['External'][key]:
            continue
        for item in dictIn['External'][key]['Tree']:
            # This will go through the list of objects:
            if isinstance(item, list):
                tempList.extend(ProcessList(item))
            elif isinstance(item, dict):
                if 'Datatype' not in item:
                    continue
                if item['Datatype'] == recordName:
                    tempList.append(item)
        if len(tempList) > 0:
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = tempList
    return newObject

def GetThisRecordMaster(dictIn, recordName):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs vertex list records.
    """
    
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    newObject = dict()
    
    def ProcessList(listObject):
        vList = []
        if not isinstance(listObject, list):
            return
        # This is a list
        for item in listObject:
            if isinstance(item, list):
                tempList = ProcessList(item)
                if len(tempList) > 0:
                    vList.extend(tempList)
            elif isinstance(item, dict):
                if 'Datatype' not in item:
                    continue
                else:
                    if item['Datatype'] == recordName:
                        vList.append(item)
        return vList
    
    # This is similar to the ExtractExternal function.
    print "Processing..."
    
    tempList = []
    for item in dictIn['Tree']:
        # This will go through the list of objects:
        if isinstance(item, list):
            tempList.extend(ProcessList(item))
        elif isinstance(item, dict):
            if 'Datatype' not in item:
                continue
            if item['Datatype'] == recordName:
                tempList.append(item)
    if len(tempList) > 0:
        newObject = tempList
    return newObject

def CreateTexturedHeaderFile(modelName, dictIn, filename = "scene.h", scale = 5):
    coordinates = VertexListToCoords(dictIn)
    if modelName not in coordinates:
        raise Exception("Unable to find model \"" + modelName + "\"")
    
    textureFiles = GetThisRecord(dictIn, "TexturePalette")[modelName]
    if len(textureFiles) > 1:
        raise NotImplemented("Unable to handle more than one type of texture file.")
    textureFiles = [file['Filename'].replace('\\', os.path.sep) for file in textureFiles]
    textureFilename = textureFiles[0][textureFiles[0].rindex(os.path.sep)+1:]
    textureCoords = VertexListToTextureCoords(dictIn)[modelName]
    # Simplify dictionary object
    coordinates = coordinates[modelName]
    
    HEADER = ["#ifndef SCENE_H_\n",
        "#define SCENE_H_\n",
        "\n",
        "#include \"scenecalcs.h\"\n",
        "#include \"renderscene.h\"\n",
        "#include <SOIL/SOIL.h>",
        "\n",
        "// Called to draw scene\n",
        "void RenderScene(void)\n",
        "{\n",
        "float normal[3];	// Storage for calculated surface normal\n",
        "\n",
        "// Clear the window with current clearing colour\n",
        "glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);\n",
        "\n",
        "glActiveTexture(GL_TEXTURE0);\n",
        "GLuint texture_id = SOIL_load_OGL_texture\n",
        "(\n",
        "\"" + textureFilename + "\",\n",
        "SOIL_LOAD_AUTO,\n",
        "SOIL_CREATE_NEW_ID,\n",
        "SOIL_FLAG_INVERT_Y\n",
        ");\n",
        "if(texture_id == 0)\n",
        "\tprintf(\"Soil loading error.\\n\");\n",
        "//\tcerr << \"SOIL loading error: '\" << SOIL_last_result() << \"' (\" << \"" + textureFilename + "\" << \")\" << endl;\n",
        "//glActiveTexture(texture_id);\n",
        "glGenTextures(1, &texture_id);\n",
        "\n",
        "glEnable(GL_TEXTURE_2D);\n",
        "\n",
        "int img_width, img_height;\n",
        "unsigned char* img = SOIL_load_image(\"" + textureFilename + "\", &img_width, &img_height, NULL, 0);\n",
        "\n",
        "glBindTexture(GL_TEXTURE_2D, texture_id);\n",
        "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);\n",
        "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);\n",
        "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);\n",
        "glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);\n",
        "glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);\n",
        "\n",
        "// Save the matrix state and do the rotations\n",
        "glPushMatrix();\n",
        "glRotatef(xRot, 1.0f, 0.0f, 0.0f);\n",
        "glRotatef(yRot, 0.0f, 1.0f, 0.0f);\n",
        "\n",
        "\n",
        "// Nose Cone /////////////////////////////\n",
        "// Set material colour\n",
        "//glColor3ub(128, 128, 128);\n",
        "glBegin(GL_TRIANGLES);\n"]
    
    FOOTER = ["glEnd();\n",
        "}\n",
        "glBindTexture(GL_TEXTURE_2D, 0);\n",
        "glDisable(GL_TEXTURE_2D);\n",
        "glDeleteTextures(1, &texture_id);\n",
        "free(img);\n",
        "\n",
        "// Restore the matrix state\n",
        "glPopMatrix();\n",
        "// Display the results\n",
        "glutSwapBuffers();\n",
        "}\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    
    for idx in range(0, coordinates.shape[0], 3):
        if idx > 0:
            outFile.write('}\n\n')
        v1, v2, v3 = scale * coordinates[idx, :]
        outFile.write("{\n")
        outFile.write("float v[3][3] = {{%ff, %ff, %ff},\n" % (v1, v2, v3)),
        
        v1, v2, v3 = scale * coordinates[idx + 1, :]
        outFile.write("     {%ff, %ff, %ff},\n" % (v1, v2, v3))
        
        v1, v2, v3 = scale * coordinates[idx + 2, :]
        outFile.write("     {%ff, %ff, %ff}};\n" % (v1, v2, v3))
        
        outFile.write("calcNormal(v, normal);\n")
        outFile.write("glNormal3fv(normal);\n")
        outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx, 0], 1-textureCoords[idx, 1]))
        outFile.write("glVertex3fv(v[0]);\n")
        outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+1, 0], 1-textureCoords[idx+1, 1]))
        outFile.write("glVertex3fv(v[1]);\n")
        outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+2, 0], 1-textureCoords[idx+2, 1]))
        outFile.write("glVertex3fv(v[2]);\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()


def VertexListToComplexTextureCoords(dictIn):
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
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]] = dict()
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]]['Coords'] = np.vstack(tempList)
            newObject[key[key.rindex(os.path.sep)+1:key.rindex('.')]]['TexturePattern'] = np.vstack(tempList2)
    return newObject

def VertexListToComplexTextureCoordsMaster(dictIn):
    """
        The input to this function should be the complete records dictionary.
        
        This function outputs a dictionary of NumPy coordinates.
    """
    if not isinstance(dictIn, dict):
        raise Exception('Input it not a dictionary object.')
    
    # This is similar to the ExtractExternal function.
    tempList = []
    tempList2 = []
    
    newObject = dict()
    
    if 'VertexList' not in dictIn:
        raise Exception("Unable to find the vertex list.")
    
    for item, txpidx in zip(dictIn['VertexList'], dictIn['TexturePatterns']):
        tempMat = None
        for idx, offset in enumerate(item):
            if tempMat is None:
                tempMat = np.zeros((len(item), max(dictIn['Vertices'][offset]['TextureCoordinate'].shape)))
            tempMat[idx, :] = dictIn['Vertices'][offset]['TextureCoordinate']
        tempList.append(tempMat)
        tempList2.append(np.ones((len(item), 1)) * txpidx)
    if len(tempList) > 0:
        newObject['Coords'] = np.vstack(tempList)
        newObject['TexturePattern'] = np.vstack(tempList2)
    return newObject


def CreateComplexTextureHeaderFile(modelName, dictIn, filename = "scene.h", scale = 5):
    if modelName is None:
        coordinates = VertexListToCoordsMaster(dictIn)
        
        textureFiles = GetThisRecordMaster(dictIn, "TexturePalette")
        tempDict = VertexListToComplexTextureCoordsMaster(dictIn)
        modelName = "Model"
    else:
        coordinates = VertexListToCoords(dictIn)
        if modelName not in coordinates:
            raise Exception("Unable to find model \"" + modelName + "\"")
        
        textureFiles = GetThisRecord(dictIn, "TexturePalette")[modelName]
        tempDict = VertexListToComplexTextureCoords(dictIn)[modelName]
        
        # Simplify dictionary object
        coordinates = coordinates[modelName]
    
    textureFilenames = [filen['Filename'].replace('\\', os.path.sep) for filen in textureFiles]
    textureIndices = [filen['TexturePatternIdx'] for filen in textureFiles]
    # Remove path and only have filename.
    tempList = []
    isSGI = []
    for filen in textureFilenames:
        # Note whether this is an SGI file
        isSGI.append(filen[-3:].lower() == "rgb" or filen[-3:].lower == 'sgi')
        # Remove path if it's there.
        if filen.count(os.path.sep) > 0:
            tempList.append(filen[filen.rindex(os.path.sep)+1:])
        else:
            tempList.append(filen)
    
    textureFilenames = tempList
    
    textureCoords = tempDict['Coords']
    
    texturePatternIdx = tempDict['TexturePattern'] - min(tempDict['TexturePattern'])
    
    HEADER = ["#ifndef SCENE_H_\n",
        "#define SCENE_H_\n",
        "\n",
        "#include \"scenecalcs.h\"\n",
        "#include \"renderscene.h\"\n",
        "#include <SOIL/SOIL.h>\n",
        "#include \"texture.h\"\n\n",
        "// This script is for model \"" + str(modelName) + "\"\n",
        "\n",
        "// Called to draw scene\n",
        "void RenderScene(void)\n",
        "{\n",
        "float normal[3];\t// Storage for calculated surface normal\n",
        "\n",
        "// Clear the window with current clearing colour\n",
        "glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);\n",
        "\n",
        "GLsizei num_textures = " + str(len(textureFilenames)) + ";\n",
        "GLuint texture_ids[num_textures];\n"]
    
    HEADERCONT =["\n",
        "\n",
        "// Save the matrix state and do the rotations\n",
        "glPushMatrix();\n",
        "glRotatef(xRot, 1.0f, 0.0f, 0.0f);\n",
        "glRotatef(yRot, 0.0f, 1.0f, 0.0f);\n",
        "glRotatef(zRot, 0.0f, 0.0f, 1.0f);\n",
        "glScalef(scale, scale, scale);\n",
        "\n",
        "\n",
        "// Nose Cone /////////////////////////////\n",
        "// Set material colour\n",
        "//glColor3ub(128, 128, 128);\n",
        "glBegin(GL_TRIANGLES);\n"]
    
    FOOTER = ["glEnd();\n",
        "}\n",
        "glBindTexture(GL_TEXTURE_2D, 0);\n",
        "glDisable(GL_TEXTURE_2D);\n",
        "glDeleteTextures(num_textures, texture_ids);\n",
        "\n",
        "// Restore the matrix state\n",
        "glPopMatrix();\n",
        "// Display the results\n",
        "glutSwapBuffers();\n",
        "}\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    tempStr = ""
    for idx, filen in enumerate(textureFilenames):
        tempStr += "// texture_ids[" + str(idx) + "] = SOIL_load_OGL_texture\n//(\n//\"" + filen + "\",\n//SOIL_LOAD_AUTO,\n//SOIL_CREATE_NEW_ID,\n//SOIL_FLAG_INVERT_Y\n//);\n"
    outFile.write(tempStr)
    
    for idx in range(len(textureFilenames)):
        outFile.write("//if(texture_ids[" + str(idx) + "] == 0)\n\t//printf(\"Soil loading error with texture " + str(idx) + ".\\n\");\n")
        # "//\tcerr << \"SOIL loading error: '\" << SOIL_last_result() << \"' (\" << \"" + textureFilename + "\" << \")\" << endl;\n",
    outFile.write("glGenTextures(num_textures, texture_ids);\n")
    
    for idx, filen in enumerate(textureFilenames):
        if isSGI[idx]:
            outFile.write("{\nint img_width, img_height, depth;\n")
            outFile.write("GLenum format = GL_RGBA;\n")
            outFile.write("glActiveTexture(GL_TEXTURE0 + " + str(idx) + ");\n",)
            outFile.write("unsigned *img = read_texture(\"" + filen + "\", &img_width, &img_height, &depth);\n\n")
            outFile.write("glEnable(GL_BLEND);\nglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);\n\n")
            outFile.write("glEnable(GL_TEXTURE_2D);\n")
            outFile.write("glBindTexture(GL_TEXTURE_2D, texture_ids[" + str(idx) + "]);\n")
            outFile.write("glPixelStorei(GL_UNPACK_ALIGNMENT, 1);\n")
            outFile.write("//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);\n")
            outFile.write("//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);\n")
            outFile.write("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);\n")
            outFile.write("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);\n")
            outFile.write("glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);\n")
            outFile.write("glTexImage2D(GL_TEXTURE_2D, 0, 3, img_width, img_height, 0, format, GL_UNSIGNED_BYTE, (GLubyte *) img);\n")
            outFile.write("free(img);\n}\n")
        else:
            outFile.write("{\nint img_width, img_height, channels;\n")
            outFile.write("GLenum format = GL_RGBA;\n")
            outFile.write("glActiveTexture(GL_TEXTURE0 + " + str(idx) + ");\n",)
            outFile.write("unsigned char* img = SOIL_load_image(\"" + filen + "\", &img_width, &img_height, &channels, SOIL_LOAD_AUTO);\n\n")
            outFile.write("if(channels == 3)\n")
            outFile.write("\tformat = GL_RGB;\n")
            outFile.write("else\n{\n")
            outFile.write("//printf(\"Channels: \%i\\n\", channels);\n")
            outFile.write("\tglEnable(GL_BLEND);\n\tglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);\n}\n")
            outFile.write("glEnable(GL_TEXTURE_2D);\n")
            outFile.write("glBindTexture(GL_TEXTURE_2D, texture_ids[" + str(idx) + "]);\n")
            # "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);\n",
            # "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);\n",
            # "//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);\n",
            outFile.write("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);\n")
            outFile.write("glTexImage2D(GL_TEXTURE_2D, 0, channels, img_width, img_height, 0, format, GL_UNSIGNED_BYTE, img);\n")
            outFile.write("SOIL_free_image_data(img);\n}\n")
    
    outFile.writelines(HEADERCONT)
    
    print "Number of triangles: " + str(coordinates.shape[0] / 3) + "\n\n"
    
    for idx in range(0, coordinates.shape[0], 3):
        if idx > 0:
            outFile.write('}\n\n')
        v1, v2, v3 = scale * coordinates[idx, :]
        outFile.write("{\n")
        outFile.write("float v[3][3] = {{%ff, %ff, %ff},\n" % (v1, v2, v3)),
        
        v1, v2, v3 = scale * coordinates[idx + 1, :]
        outFile.write("     {%ff, %ff, %ff},\n" % (v1, v2, v3))
        
        v1, v2, v3 = scale * coordinates[idx + 2, :]
        outFile.write("     {%ff, %ff, %ff}};\n" % (v1, v2, v3))
        
        outFile.write("calcNormal(v, normal);\n")
        outFile.write("glActiveTexture(GL_TEXTURE0 + " + str(int(texturePatternIdx[idx, 0])) + ");\n")
        outFile.write("glBindTexture(GL_TEXTURE_2D, texture_ids[" + str(int(texturePatternIdx[idx, 0])) + "]);\n")
        
        outFile.write("glNormal3fv(normal);\n")
        if isSGI[int(texturePatternIdx[idx, 0])]:
            # This is an SGI file and was opened with the SGI file container. Do not invert y.
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx, 0], textureCoords[idx, 1]))
            outFile.write("glVertex3fv(v[0]);\n")
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+1, 0], textureCoords[idx+1, 1]))
            outFile.write("glVertex3fv(v[1]);\n")
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+2, 0], textureCoords[idx+2, 1]))
            outFile.write("glVertex3fv(v[2]);\n")        
        else:
            # This was opened with SOIL, so invert y.
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx, 0], 1-textureCoords[idx, 1]))
            outFile.write("glVertex3fv(v[0]);\n")
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+1, 0], 1-textureCoords[idx+1, 1]))
            outFile.write("glVertex3fv(v[1]);\n")
            outFile.write("glTexCoord2d(%ff, %ff);\n" % (textureCoords[idx+2, 0], 1-textureCoords[idx+2, 1]))
            outFile.write("glVertex3fv(v[2]);\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()

def CreateComplexRTHeaderFile(modelName, dictIn, filename = "OFconstruct.h", scale = 5):
    if modelName is None:
        coordinates = VertexListToCoordsMaster(dictIn)
        modelName = "Model"
    else:
        coordinates = VertexListToCoords(dictIn)
        if modelName not in coordinates:
            raise Exception("Unable to find model \"" + modelName + "\"")
        
        # Simplify dictionary object
        coordinates = coordinates[modelName]
    
    HEADER = ["#ifndef OFCONSTRUCT_H_\n",
        "#define OFCONSTRUCT_H_\n",
        "\n",
        "#include \"fpmath.h\"\n",
        "#include \"craytracer.h\"\n",
        "#include \"datatypes.h\"\n",
        "#include \"rays.h\"\n",
        "#include \"image.h\"\n",
        "#include \"lighting.h\"\n",
        "#include \"objects.h\"\n",
        "#include \"colours.h\"\n",
        "#include \"shapes.h\"\n",
        "#include \"mathstats.h\"\n",
        "#include \"funcstats.h\"\n\n",
        "// This script is for model \"" + str(modelName) + "\"\n",
        "\n",
        "// Put the object(s) on the scene\n",
        "void populateScene(Scene *scene, Light lightSrc, MathStat *m, FuncStat *f)\n",
        "{\n",
        "//fixedp normal[3];\t// Storage for calculated surface normal\n",
        "\n"]
    
    FOOTER = ["Vector draw(Ray ray, Scene scene, Light light, int recursion, MathStat *m, FuncStat *f)\n",
        "{\n",
        "    Hit hit;\n",
        "    Vector outputColour, reflectiveColour, refractiveColour, textureColour;\n",
        "    fixedp reflection, refraction;\n",
        "    \n",
        "    (*f).draw++;\n",
        "    \n",
        "    // Default is black. We can add to this (if there's a hit) \n",
        "    // or just return it (if there's no object)\n",
        "    setVector(&outputColour, 0, 0, 0, f);\n",
        "    \n",
        "    hit = sceneIntersection(ray, scene, m, f);\n",
        "    \n",
        "    // Determine whether there was a hit. Otherwise default.\n",
        "    if (hit.objectIndex >= 0)\n",
        "    {\n",
        "        // There was a hit.\n",
        "        Vector lightDirection = vecNormalised(vecSub(light.location, hit.location, m, f), m, f);\n",
        "        \n",
        "        // Determine whether this has a texture or not\n",
        "        if (scene.object[hit.objectIndex].material.textureIdx < 0)\n",
        "            setVector(&textureColour, -1, -1, -1, f);\n",
        "        else\n",
        "            textureColour = getColour(Textures[scene.object[hit.objectIndex].material.textureIdx], scene, hit, m, f);\n\n",
        "        // outputColour = vecAdd(ambiance(hit, scene, light, m, f), diffusion(hit, scene, light, m, f), m, f);\n",
        "        outputColour = vecAdd(ambiance(hit, scene, light, textureColour, m, f), vecAdd(diffusion(hit, scene, light, lightDirection, textureColour, m, f), specular(hit, scene, light, lightDirection, textureColour, m, f), m, f), m, f);\n",
        "        \n",
        "        // Should we go deeper?\n",
        "        if (recursion > 0)\n",
        "        {\n",
        "            // Yes, we should\n",
        "            // Get the reflection\n",
        "            reflectiveColour = draw(reflectRay(hit, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            reflection = scene.object[hit.objectIndex].material.reflectivity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(reflection, reflectiveColour, m, f), m, f);\n",
        "            \n",
        "            // Get the refraction\n",
        "            refractiveColour = draw(refractRay(hit, scene.object[hit.objectIndex].material.inverserefractivity, scene.object[hit.objectIndex].material.squareinverserefractivity, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            refraction = scene.object[hit.objectIndex].material.opacity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(refraction, refractiveColour, m, f), m, f);\n",
        "        }\n",
        "        \n",
        "        // We've got what we needed after the hit, so return\n",
        "        statSubtractFlt(m, 1);\n",
        "        return scalarVecMult(fp_fp1 - traceShadow(hit, scene, light, lightDirection, m, f), outputColour, m, f);\n",
        "    }\n",
        "    \n",
        "    // No hit, return black.\n",
        "    \n",
        "    return outputColour;\n",
        "}\n",
        "\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    
    noTriangles = coordinates.shape[0] / 3
    
    print "Number of triangles: " + str(noTriangles) + "\n\n"
    
    outFile.write("Object myObj;\nMaterial myMat;\nVector red = int2Vector(RED);\n")
    outFile.write("Vector u, v, w;\n\n")
    outFile.write("setMaterial(&myMat, lightSrc, red, fp_Flt2FP(0.0), fp_Flt2FP(0.5), fp_Flt2FP(0.0), fp_Flt2FP(0.0), fp_Flt2FP(0.0), fp_Flt2FP(0.8), fp_Flt2FP(1.4), -1, m, f);\n");
    outFile.write("Triangle *triangle;\n")
    outFile.write("triangle = (Triangle *)malloc(sizeof(Triangle) * " + str(noTriangles) + ");\n")
    outFile.write("// Now begin object writing\n\n")
    
    for idx in range(0, coordinates.shape[0], 3):
        outFile.write("// Triangle " + str(idx / 3) + ":\n\n")
        
        v1, v2, v3 = scale * coordinates[idx, :]
        outFile.write("setVector(&u, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
        
        v1, v2, v3 = scale * coordinates[idx + 1, :]
        outFile.write("setVector(&v, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
        
        v1, v2, v3 = scale * coordinates[idx + 2, :]
        outFile.write("setVector(&w, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
        
        outFile.write("setTriangle(&triangle[" + str(idx / 3) + "], u, v, w, m, f);\n\n")
    
    outFile.write("setObject(&myObj, myMat, " + str(noTriangles) + ", triangle, f);\n")
    outFile.write("transformObject(&myObj, matMult(genTransMatrix(fp_Flt2FP(1.), fp_Flt2FP(-5.), -fp_Flt2FP(30.), m, f), genZRotateMat(-fp_Flt2FP(20.), m, f), m, f), m, f);\n")
    outFile.write("initialiseScene(scene, 1, f);\n")
    outFile.write("addObject(scene, myObj, f);\n")
    
    outFile.write("}\n\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()

def CreateComplexTextureRTHeaderFile(modelName, dictIn, filename = "OFconstruct.h", scale = 5):
    if modelName is None:
        coordinates = VertexListToCoordsMaster(dictIn)
        modelName = "Model"
        textureFiles = GetThisRecordMaster(dictIn, "TexturePalette")
        tempDict = VertexListToComplexTextureCoordsMaster(dictIn)
    else:
        coordinates = VertexListToCoords(dictIn)
        if modelName not in coordinates:
            raise Exception("Unable to find model \"" + modelName + "\"")
        
        tempDict = VertexListToComplexTextureCoords(dictIn)[modelName]
        
        textureFiles = GetThisRecord(dictIn, "TexturePalette")[modelName]
        
        # Simplify dictionary object
        coordinates = coordinates[modelName]
    
    textureFilenames = [filen['Filename'].replace('\\', os.path.sep) for filen in textureFiles]
    textureIndices = [filen['TexturePatternIdx'] for filen in textureFiles]
    # Remove path and only have filename.
    tempList = []
    for filen in textureFilenames:
        # Remove path if it's there.
        if filen.count(os.path.sep) > 0:
            tempList.append(filen[filen.rindex(os.path.sep)+1:])
        else:
            tempList.append(filen)
    
    textureFilenames = tempList
    
    # Now obtain unique filenames
    uniqueTextureFilenames = []
    for filen in textureFilenames:
        if filen not in uniqueTextureFilenames:
            uniqueTextureFilenames.append(filen)
    
    textureCoords = tempDict['Coords']
    
    texturePatternIdx = tempDict['TexturePattern'] - min(tempDict['TexturePattern'])
    
    HEADER = ["#ifndef OFCONSTRUCT_H_\n",
        "#define OFCONSTRUCT_H_\n",
        "\n",
        "#include \"fpmath.h\"\n",
        "#include \"craytracer.h\"\n",
        "#include \"datatypes.h\"\n",
        "#include \"rays.h\"\n",
        "#include \"image.h\"\n",
        "#include \"lighting.h\"\n",
        "#include \"objects.h\"\n",
        "#include \"colours.h\"\n",
        "#include \"shapes.h\"\n",
        "#include \"mathstats.h\"\n",
        "#include \"funcstats.h\"\n",
        "#include \"textures.h\"\n\n",
        "// This script is for model \"" + str(modelName) + "\"\n\n",
        "Texture Textures[" + str(len(uniqueTextureFilenames)) + "];\n"
        "\n",
        "// Put the object(s) on the scene\n",
        "void populateScene(Scene *scene, Light lightSrc, MathStat *m, FuncStat *f)\n",
        "{\n",
        "    //fixedp normal[3];\t// Storage for calculated surface normal\n",
        "    \n"]
    
    FOOTER = ["Vector draw(Ray ray, Scene scene, Light light, int recursion, MathStat *m, FuncStat *f)\n",
        "{\n",
        "    Hit hit;\n",
        "    Vector outputColour, reflectiveColour, refractiveColour, textureColour;\n",
        "    fixedp reflection, refraction;\n",
        "    \n",
        "    (*f).draw++;\n",
        "    \n",
        "    // Default is black. We can add to this (if there's a hit) \n",
        "    // or just return it (if there's no object)\n",
        "    setVector(&outputColour, 0, 0, 0, f);\n",
        "    \n",
        "    hit = sceneIntersection(ray, scene, m, f);\n",
        "    \n",
        "    // Determine whether there was a hit. Otherwise default.\n",
        "    if (hit.objectIndex >= 0)\n",
        "    {\n",
        "        // There was a hit.\n",
        "        Vector lightDirection = vecNormalised(vecSub(light.location, hit.location, m, f), m, f);\n",
        "        \n",
        "        // Determine whether this has a texture or not\n",
        "        if (scene.object[hit.objectIndex].material.textureIdx < 0)\n",
        "            setVector(&textureColour, -1, -1, -1, f);\n",
        "        else\n",
        "            textureColour = getColour(Textures[scene.object[hit.objectIndex].material.textureIdx], scene, hit, m, f);\n\n",
        "        // outputColour = vecAdd(ambiance(hit, scene, light, m, f), diffusion(hit, scene, light, m, f), m, f);\n",
        "        outputColour = vecAdd(ambiance(hit, scene, light, textureColour, m, f), vecAdd(diffusion(hit, scene, light, lightDirection, textureColour, m, f), specular(hit, scene, light, lightDirection, textureColour, m, f), m, f), m, f);\n",
        "        \n",
        "        // Should we go deeper?\n",
        "        if (recursion > 0)\n",
        "        {\n",
        "            // Yes, we should\n",
        "            // Get the reflection\n",
        "            reflectiveColour = draw(reflectRay(hit, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            reflection = scene.object[hit.objectIndex].material.reflectivity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(reflection, reflectiveColour, m, f), m, f);\n",
        "            \n",
        "            // Get the refraction\n",
        "            refractiveColour = draw(refractRay(hit, scene.object[hit.objectIndex].material.inverserefractivity, scene.object[hit.objectIndex].material.squareinverserefractivity, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            refraction = scene.object[hit.objectIndex].material.opacity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(refraction, refractiveColour, m, f), m, f);\n",
        "        }\n",
        "        \n",
        "        // We've got what we needed after the hit, so return\n",
        "        statSubtractFlt(m, 1);\n",
        "        return scalarVecMult(fp_fp1 - traceShadow(hit, scene, light, lightDirection, m, f), outputColour, m, f);\n",
        "    }\n",
        "    \n",
        "    // No hit, return black.\n",
        "    \n",
        "    return outputColour;\n",
        "}\n",
        "\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    
    # Scale coordinates all at once:
    coordinates *= scale
    
    noTriangles = coordinates.shape[0] / 3
    
    print "Number of triangles: " + str(noTriangles) + "\n\n"
    print "Number of textures: " + str(len(textureFilenames)) + "\n\n"
    
    outFile.write("    Object myObj;\n")
    outFile.write("    Material myMat[" + str(len(textureFilenames)) + "];\n")
    outFile.write("    Vector lgrey = int2Vector(LIGHT_GREY);\n")
    outFile.write("    Vector u, v, w;\n\n")
    outFile.write("    UVCoord uUV, vUV, wUV;\n\n")
    outFile.write("    initialiseScene(scene, " + str(len(textureFilenames)) + ", f);\n")
    outFile.write("    Triangle *triangle;\n")
    # Import unique textures first:
    for textIdx, fn in enumerate(uniqueTextureFilenames):
        outFile.write("    ReadTexture(&Textures[" + str(textIdx) + "],\"" + fn[:-3] + "tga\", f);\n")
    # Now import materials and look up the appropriate texture based on the filename
    for textIdx, fn in enumerate(textureFilenames):
        outFile.write("    setMaterial(&myMat[" + str(textIdx) + "], lightSrc, lgrey, fp_Flt2FP(1.0), 0, fp_Flt2FP(0.1), fp_Flt2FP(0.5), fp_Flt2FP(0.2), 0, fp_Flt2FP(1.4), " + str(uniqueTextureFilenames.index(fn)) + ", m, f);\n")
        
        # Now retrieve the appropriate data for this texture.
        theseCoordinates = coordinates[texturePatternIdx.flatten() == textIdx, :]
        
        # Get the number of triangles in this group:
        noTrianglesSubset = theseCoordinates.shape[0] / 3
        
        # Get the subset for texture coordinates:
        theseTextureCoords = textureCoords[texturePatternIdx.flatten() == textIdx, :]
        
        outFile.write("    // Texture " + str(textIdx) + "\n\n")
        outFile.write("    triangle = (Triangle *)malloc(sizeof(Triangle) * " + str(noTrianglesSubset) + ");\n")
        outFile.write("    // Now begin object writing\n\n")
    
        for idx in range(0, theseCoordinates.shape[0], 3):
            outFile.write("    // Triangle " + str(idx / 3) + ":\n\n")
        
            v1, v2, v3 = theseCoordinates[idx, :]
            outFile.write("    setVector(&u, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
            outFile.write("    setUVCoord(&uUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (theseTextureCoords[idx, 0], theseTextureCoords[idx, 1]))
        
            v1, v2, v3 = theseCoordinates[idx + 1, :]
            outFile.write("    setVector(&v, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
            outFile.write("    setUVCoord(&vUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (theseTextureCoords[idx + 1, 0], theseTextureCoords[idx + 1, 1]))
        
            v1, v2, v3 = theseCoordinates[idx + 2, :]
            outFile.write("    setVector(&w, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
            outFile.write("    setUVCoord(&wUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (theseTextureCoords[idx + 2, 0], theseTextureCoords[idx + 2, 1]))
        
            outFile.write("    setUVTriangle(&triangle[" + str(idx / 3) + "], u, v, w, uUV, vUV, wUV, m, f);\n\n")
    
        outFile.write("    setObject(&myObj, myMat[" + str(textIdx) + "], " + str(noTrianglesSubset) + ", triangle, f);\n")
        outFile.write("    transformObject(&myObj, matMult(genTransMatrix(fp_Flt2FP(1.), fp_Flt2FP(-5.), -fp_Flt2FP(15.), m, f), matMult(genYRotateMat(fp_Flt2FP(160.), m, f), genXRotateMat(fp_Flt2FP(-90.), m, f), m, f), m, f), m, f);\n")
        outFile.write("    addObject(scene, myObj, f);\n")
    
    outFile.write("}\n\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()


def CreateWorldHeaderFile(dictIn, filename = "OFconstruct.h"):
    """
    CreateWorldHeaderFile imports the contents of a dictionary object which may link to
    multiple models. This function collates this information and saves it to a C header
    file which can be imported directly into the ray tracer.
    
    """
    
    # Let's go on the initial assumption that this record consists of external records
    # which need to be called and each external record may be subjected to a matrix
    # transformation which occurs after the external record is declared.
    # This requires us to navigate through the record list and pull out relevant records
    # and substituting external records for the "External" equivalent, translating or
    # transforming those coordinates as defined by a possible matrix after the record.
    
    
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
    
    # Let's start by opening the output file and adding the header. 
    
    
    
    # if modelName is None:
    #     coordinates = VertexListToCoordsMaster(dictIn)
    #     modelName = "Model"
    #     textureFiles = GetThisRecordMaster(dictIn, "TexturePalette")
    #     tempDict = VertexListToComplexTextureCoordsMaster(dictIn)
    # else:
    #     coordinates = VertexListToCoords(dictIn)
    #     if modelName not in coordinates:
    #         raise Exception("Unable to find model \"" + modelName + "\"")
    #     
    #     tempDict = VertexListToComplexTextureCoords(dictIn)[modelName]
    #     
    #     textureFiles = GetThisRecord(dictIn, "TexturePalette")[modelName]
    #     
    #     # Simplify dictionary object
    #     coordinates = coordinates[modelName]
    #
    # texturePatternIdx = tempDict['TexturePattern'] - min(tempDict['TexturePattern'])
    
    HEADER = ["#ifndef OFCONSTRUCT_H_\n",
        "#define OFCONSTRUCT_H_\n",
        "\n",
        "#include \"fpmath.h\"\n",
        "#include \"craytracer.h\"\n",
        "#include \"datatypes.h\"\n",
        "#include \"rays.h\"\n",
        "#include \"image.h\"\n",
        "#include \"lighting.h\"\n",
        "#include \"objects.h\"\n",
        "#include \"colours.h\"\n",
        "#include \"shapes.h\"\n",
        "#include \"mathstats.h\"\n",
        "#include \"funcstats.h\"\n",
        "#include \"textures.h\"\n\n",
        "// This script is for an entire scene.\n\n",
        "Texture Textures[" + str(len(PrimaryTextureFilenames)) + "];\n"
        "\n",
        "// Put the object(s) on the scene\n",
        "void populateScene(Scene *scene, Light lightSrc, MathStat *m, FuncStat *f)\n",
        "{\n",
        "    //fixedp normal[3];\t// Storage for calculated surface normal\n",
        "    \n"]
    
    FOOTER = ["Vector draw(Ray ray, Scene scene, Light light, int recursion, MathStat *m, FuncStat *f)\n",
        "{\n",
        "    Hit hit;\n",
        "    Vector outputColour, reflectiveColour, refractiveColour, textureColour;\n",
        "    fixedp reflection, refraction;\n",
        "    \n",
        "    (*f).draw++;\n",
        "    \n",
        "    // Default is black. We can add to this (if there's a hit) \n",
        "    // or just return it (if there's no object)\n",
        "    setVector(&outputColour, 0, 0, 0, f);\n",
        "    \n",
        "    hit = sceneIntersection(ray, scene, m, f);\n",
        "    \n",
        "    // Determine whether there was a hit. Otherwise default.\n",
        "    if (hit.objectIndex >= 0)\n",
        "    {\n",
        "        // There was a hit.\n",
        "        Vector lightDirection = vecNormalised(vecSub(light.location, hit.location, m, f), m, f);\n",
        "        \n",
        "        // Determine whether this has a texture or not\n",
        "        if (scene.object[hit.objectIndex].material.textureIdx < 0)\n",
        "            setVector(&textureColour, -1, -1, -1, f);\n",
        "        else\n",
        "            textureColour = getColour(Textures[scene.object[hit.objectIndex].material.textureIdx], scene, hit, m, f);\n\n",
        "        // outputColour = vecAdd(ambiance(hit, scene, light, m, f), diffusion(hit, scene, light, m, f), m, f);\n",
        "        outputColour = vecAdd(ambiance(hit, scene, light, textureColour, m, f), vecAdd(diffusion(hit, scene, light, lightDirection, textureColour, m, f), specular(hit, scene, light, lightDirection, textureColour, m, f), m, f), m, f);\n",
        "        \n",
        "        // Should we go deeper?\n",
        "        if (recursion > 0)\n",
        "        {\n",
        "            // Yes, we should\n",
        "            // Get the reflection\n",
        "            reflectiveColour = draw(reflectRay(hit, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            reflection = scene.object[hit.objectIndex].material.reflectivity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(reflection, reflectiveColour, m, f), m, f);\n",
        "            \n",
        "            // Get the refraction\n",
        "            refractiveColour = draw(refractRay(hit, scene.object[hit.objectIndex].material.inverserefractivity, scene.object[hit.objectIndex].material.squareinverserefractivity, m, f), scene, light, recursion - 1, m, f);\n",
        "            statSubtractInt(m, 1);\n",
        "            refraction = scene.object[hit.objectIndex].material.opacity;\n",
        "            outputColour = vecAdd(outputColour, scalarVecMult(refraction, refractiveColour, m, f), m, f);\n",
        "        }\n",
        "        \n",
        "        // We've got what we needed after the hit, so return\n",
        "        statSubtractFlt(m, 1);\n",
        "        return scalarVecMult(fp_fp1 - traceShadow(hit, scene, light, lightDirection, m, f), outputColour, m, f);\n",
        "    }\n",
        "    \n",
        "    // No hit, return black.\n",
        "    \n",
        "    return outputColour;\n",
        "}\n",
        "\n",
        "#endif\n"]
    
    outFile = open(filename, 'w')
    outFile.writelines(HEADER)
    
    outFile.write("    Object myObj;\n")
    outFile.write("    Material myMat[" + str(numberOfMaterials) + "];\n")
    outFile.write("    Vector lgrey = int2Vector(LIGHT_GREY);\n")
    outFile.write("    Vector u, v, w;\n\n")
    outFile.write("    UVCoord uUV, vUV, wUV;\n\n")
    outFile.write("    initialiseScene(scene, " + str(numberOfMaterials) + ", f);\n")
    outFile.write("    Triangle *triangle;\n")
    
    # Now load up all the necessary texture files:
    for textIdx, fn in enumerate(PrimaryTextureFilenames):
        outFile.write("    ReadTexture(&Textures[" + str(textIdx) + "],\"" + fn[:-3] + "tga\", f);\n")
    # And, whilst here, let's set up the materials cache:
    for materialPath in materialDB:
        for curIdx, textureIdx in zip(materialDB[materialPath], textureDB[materialPath]):
            outFile.write("    setMaterial(&myMat[" + str(curIdx) + "], lightSrc, lgrey, fp_Flt2FP(1.0), 0, fp_Flt2FP(0.1), fp_Flt2FP(0.5), fp_Flt2FP(0.2), 0, fp_Flt2FP(1.4), " + str(textureIdx) + ", m, f);\n")
    
    print "\n"
    
    # This part goes through all records, adding them and translating the contents to C.
    # This will link all the necessary textures too.
    for niceFilename in TransformationList:
        # Process this transformation matrix. This means converting the 4 x 4 to a 4 x 3:
        print "\rAdding model: " + niceFilename[niceFilename.rindex(os.path.sep)+1:niceFilename.rindex('.')],
        
        theseCoordinates = np.hstack((coordinates[niceFilename], np.ones((coordinates[niceFilename].shape[0], 1))))
        theseUVCoords = textureCoordinates[niceFilename]['Coords']
        thesePatterns = textureCoordinates[niceFilename]['TexturePattern'] - np.min(textureCoordinates[niceFilename]['TexturePattern'].flatten())
        
        groupedCoords = np.zeros((0, 3))
        groupedPatterns = np.zeros((0, 1))
        groupedUVCoords = np.zeros((0, 2))
        
        # Extract the transformation matrix
        for transIdx, transMat in enumerate(TransformationList[niceFilename]):
            if transMat is not None:                
                transMat = np.vstack((transMat[:3, :3].T, transMat[3, :3]))
            else:
                transMat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
            transCoords = np.dot(theseCoordinates, transMat)
            
            groupedCoords = np.vstack([groupedCoords, transCoords])
            groupedPatterns = np.vstack([groupedPatterns, thesePatterns])
            groupedUVCoords = np.vstack([groupedUVCoords, theseUVCoords])
        
        # Now we have all the information necessary. Time to go through the pattern indices
        for textIdx, matIdx in enumerate(materialDB[niceFilename]):
            # Obtain the subset of coordinates based on the pattern index
            coordSubset = groupedCoords[groupedPatterns.flatten() == textIdx, :]
            UVSubset = groupedUVCoords[groupedPatterns.flatten() == textIdx, :]
            # Compute how many triangles we're dealing with
            noTrianglesSubset = coordSubset.shape[0] / 3
            outFile.write("    triangle = (Triangle *)malloc(sizeof(Triangle) * " + str(noTrianglesSubset) + ");\n")
            for idx in range(0, coordSubset.shape[0], 3):
                v1, v2, v3 = coordSubset[idx, :]
                outFile.write("    setVector(&u, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
                outFile.write("    setUVCoord(&uUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (UVSubset[idx, 0], UVSubset[idx, 1]))
                
                v1, v2, v3 = coordSubset[idx + 1, :]
                outFile.write("    setVector(&v, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
                outFile.write("    setUVCoord(&vUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (UVSubset[idx + 1, 0], UVSubset[idx + 1, 1]))
                
                v1, v2, v3 = coordSubset[idx + 2, :]
                outFile.write("    setVector(&w, fp_Flt2FP(%ff), fp_Flt2FP(%ff), fp_Flt2FP(%ff), f);\n" % (v1, v2, v3))
                outFile.write("    setUVCoord(&wUV, fp_Flt2FP(%ff), fp_Flt2FP(%ff));\n" % (UVSubset[idx + 2, 0], UVSubset[idx + 2, 1]))
                
                outFile.write("    setUVTriangle(&triangle[" + str(idx / 3) + "], u, v, w, uUV, vUV, wUV, m, f);\n\n")
            # Triangle writing is complete. Now set the object, etc.
            outFile.write("    setObject(&myObj, myMat[" + str(matIdx) + "], " + str(noTrianglesSubset) + ", triangle, f);\n\n")
            outFile.write("    addObject(scene, myObj, f);\n")
    print("\rFinished adding models.\n")
    
    outFile.write("}\n\n")
    # Lastly, write footer
    outFile.writelines(FOOTER)
    outFile.close()
