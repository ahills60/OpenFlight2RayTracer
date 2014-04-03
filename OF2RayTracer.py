"""
This file automates the process of world extraction from OpenFlight files.

This supersedes the Extraction.py file. Functionality that's been brought
across with this includes the extraction of models (including models that
reference other models, etc.) and the writing of byte code that is 
interpreted by the C Ray tracer.

 Author: Andrew Hills (a.hills@sheffield.ac.uk)
Version: 0.0.1 (01/04/2014)
"""

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
    
    
    
def ByteCodeWriter(listin, filename="World.crt"):
    """
    ByteCodeWriter converts the input list to byte-based codes that can be read
    by the C Ray Tracer.
    
    This function creates a file (default: World.crt).
    """
    
    pass