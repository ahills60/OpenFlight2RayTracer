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
        
        for item in dictIn['External'][key]['VertexList']:
            tempMat = None
            for idx, offset in enumerate(item):
                if tempMat is None:
                    tempMat = np.zeros((len(item), max(dictIn['External'][key]['Vertices'][offset]['Coordinate'].shape)))
                tempMat[idx, :] = dictIn['External'][key]['Vertices'][offset]['Coordinate']
            tempList.append(tempMat)
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
            outFile.write("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);\n")
            outFile.write("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);\n")
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