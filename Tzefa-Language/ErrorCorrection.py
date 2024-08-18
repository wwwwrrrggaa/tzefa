from fast_edit_distance import edit_distance

import Number2Name


def giveinstructions():
    ### returns instructions for each function in the language for topy
    return listfunctions, listezfunc


listofindents = []


def updatesizelistofindnets(size):
    listofindents = [0] * (size + 1)


def tosimple(func):
    simpler = ['a', 'b', 'c', 'd']
    simpler[0] = func[0]
    if (func[1].startswith("NEW")):
        simpler[1] = 0
    else:
        simpler[1] = 1
    i = 1
    j = 2
    if (func[i].endswith("INT")):
        simpler[j] = 0
    elif (func[i].endswith("STR")):
        simpler[j] = 1
    elif (func[i].endswith("LIST")):
        simpler[j] = 2
    elif (func[i].endswith("BOOL")):
        simpler[j] = 3
    elif (func[i].endswith("COND")):
        simpler[j] = 4
    elif (func[i].endswith("STATE")):
        simpler[j] = 5
    elif (func[i].endswith("TYPE")):
        simpler[j] = 6
    elif (func[i].endswith("FUNC")):
        simpler[j] = 7
    elif (func[i].endswith("TRUTH")):
        simpler[j] = 8
    elif (func[i].endswith("COMPARE")):
        simpler[j] = 9
    elif (func[i].endswith("NUM")):
        simpler[j] = 10
    elif (func[i].endswith("TEXT")):
        simpler[j] = 11
    i = 2
    j = 3
    if (func[i].endswith("INT")):
        simpler[j] = 0
    elif (func[i].endswith("STR")):
        simpler[j] = 1
    elif (func[i].endswith("LIST")):
        simpler[j] = 2
    elif (func[i].endswith("BOOL")):
        simpler[j] = 3
    elif (func[i].endswith("COND")):
        simpler[j] = 4
    elif (func[i].endswith("STATE")):
        simpler[j] = 5
    elif (func[i].endswith("TYPE")):
        simpler[j] = 6
    elif (func[i].endswith("FUNC")):
        simpler[j] = 7
    elif (func[i].endswith("TRUTH")):
        simpler[j] = 8
    elif (func[i].endswith("COMPARE")):
        simpler[j] = 9
    elif (func[i].endswith("NUM")):
        simpler[j] = 10
    elif (func[i].endswith("TEXT")):
        simpler[j] = 11
    simpler.append(0)
    return simpler


inserver = True
listfunctions = [["MAKEINTEGER", "NEWINT", "NUM"],
                 ["MAKEBOOLEN", "NEWBOOL", "TRUTH"],
                 ["MAKESTR", "NEWSTR", "TEXT"],
                 ["NEWLIST", "NEWLIST", "INT"],
                 ["BASICCONDITION", "NEWCOND", "COMPARE"],
                 ["LEFTSIDE", "COND", "INT"],
                 ["RIGHTSIDE", "COND", "INT"],
                 ["CHANGECOMPARE", "COND", "COMPARE"],
                 ["WHILE", "COND", "NUM"],
                 ["COMPARE", "COND", "NUM"],
                 ["ELSECOMPARE", "COND", "NUM"],
                 ["ITERATE", "LIST", "NUM"],
                 ["WHILETRUE", "BOOL", "NUM"],
                 ["IFTRUE", "BOOL", "NUM"],
                 ["ELSEIF", "BOOL", "NUM"],
                 ["INTEGERFUNCTION", "NEWFUNC", "TYPE"],
                 ["STRINGFUNCTION", "NEWFUNC", "TYPE"],
                 ["LISTFUNCTION", "NEWFUNC", "TYPE"],
                 ["RETURN", "VALUE", "STATE"],
                 ["PRINTSTRING", "STR", "STATE"],
                 ["PRINTINTEGER", "INT", "STATE"],
                 ["SETINDEX", "LIST", "INT"],
                 ["TYPETOINT", "STR", "INT"],
                 ["GETSTRING", "LIST", "STR"],
                 ["GETINTEGER", "LIST", "INT"],
                 ["WRITEINTEGER", "LIST", "INT"],
                 ["WRITESTRING", "LIST", "STR"],
                 ["WRITEBOOL", "LIST", "BOOL"],
                 ["WRITELIST", "LIST", "LIST"],
                 ["GETLIST", "LIST", "LIST"],
                 ["GETBOOL", "LIST", "BOOL"],
                 ["GETTYPE", "LIST", "STR"],
                 ["LENGTH", "LIST", "INT"],
                 ["ADDVALUES", "INT", "INT"],
                 ["MULTIPLY", "INT", "INT"],
                 ["MATHPOW", "INT", "INT"],
                 ["DIVIDE", "INT", "INT"],
                 ["SIMPLEDIVIDE", "INT", "INT"],
                 ["SUBTRACT", "INT", "INT"],
                 ["MODULO", "INT", "INT"],
                 ["COMBINE", "STR", "STR"],
                 ["BLANKSPACES", "STR", "NUM"],
                 ['ADDSIZE', 'LIST', 'INT'],
                 ["ASSSIGNINT", "INT", "INT"],
                 ["STRINGASSIGN", "STR", "STR"],
                 ["COPYLIST", "LIST", "LIST"]]
listsimplefunc = [tosimple(i) for i in listfunctions]
listofindents = []
listezfunc = [i[0] for i in listfunctions]
listintegers = ["TEMPORARY", 'LOCALINT', 'LOOPINTEGER']
for i in range(101):
    listintegers.append(Number2Name.get_name(i))
liststrings = ["TEMPSTRING", "GLOBALSTR", 'LOOPSTRING', "INTEGER", "STRING", "LIST", "BOOLEAN"]
listlists = ["GLOBALLIST", "LOOPLIST"]
listconds = ['THETRUTH']
listbools = ["LOOPBOOL"]
liststate = ["STAY", "BREAK"]
listype = ["INTEGER", "STRING", "LIST", "BOOLEAN"]
lookuptype = {"INTEGER": "INT", "STRING": "STR", "LIST": "LIST", "BOOLEAN": "BOOL"}
listtruth = ["TRUE", "FALSE"]
listcompare = ["EQUALS", "BIGEQUALS", "BIGGER"]
listall = [listintegers, liststrings, listlists, listbools, listconds, liststate, listype, listezfunc, listtruth,
           listcompare]
thetype = []
insidefunction = False
counter = 0


def getsimples():
    return listsimplefunc


def sendlines(i):
    global listofindents
    listofindents = [0 for b in range(i)]


def giveindents():
    return listofindents


def findword(somelist, word):
    min = 999
    tobereturned = [0, 0,
                    0]
    lentobereturned = 16
    for b in range(len(somelist)):
        lenword = len(word)
        i = somelist[b]
        lenofi = len(i)
        if (i == word):
            return [i, b]
        else:
            distance = edit_distance(word, i, 4)
            if (distance < min):
                min = distance
                tobereturned = [i, b]
                lentobereturned = len(tobereturned[0])
            elif (distance == min):
                if (abs(lenword - lenofi) < abs(lenword - lentobereturned)):
                    tobereturned = [i, b]
                    lentobereturned = len(tobereturned[0])

    return tobereturned


def handelfirstword(firstword):
    func, index = findword(listezfunc, firstword)
    if (listfunctions[index][2] != "NUM"):
        return (func, index, 0)
    else:
        return (func, index, 1)


def toline(line, index, listofindents):
    global counter
    global thetype
    global insidefunction
    disthreeline = line.split(" ")
    threeline = ["", "", ""]
    problem = False
    threeline[0] = listezfunc[index]
    simpler = listsimplefunc[index]
    if (threeline[0] == "STRINGFUNCTION"):
        if (insidefunction):
            problem = True
        else:
            insidefunction = True
            threeline[1] = disthreeline[1]
            threeline[2] = findword(listype, disthreeline[2])[0]
            thetype.append(threeline[2])
            newsomething = [threeline[1], "STR", lookuptype[threeline[2]]]
            listezfunc.append(newsomething[0])
            listfunctions.append(newsomething)
            simplerer = tosimple(newsomething)
            listsimplefunc.append(simplerer)
    elif (threeline[0] == "INTEGERFUNCTION"):
        if (insidefunction):
            problem = True
        else:
            insidefunction = True
            threeline[1] = disthreeline[1]
            threeline[2] = findword(listype, disthreeline[2])[0]
            thetype.append(threeline[2])
            newsomething = [threeline[1], "INT", lookuptype[threeline[2]]]
            listezfunc.append(newsomething[0])
            listfunctions.append(newsomething)
            simplerer = tosimple(newsomething)
            listsimplefunc.append(simplerer)
    elif (threeline[0] == "LISTFUNCTION"):
        if (insidefunction):
            problem = True
        else:
            insidefunction = True
            threeline[1] = disthreeline[1]
            threeline[2] = findword(listype, disthreeline[2])[0]
            thetype.append(threeline[2])
            newsomething = [threeline[1], "LIST", lookuptype[threeline[2]]]
            listezfunc.append(newsomething[0])
            listfunctions.append(newsomething)
            simplerer = tosimple(newsomething)
            listsimplefunc.append(simplerer)
    elif (simpler[0] == "RETURN"):
        if (len(thetype) == 0):
            problem = True
        else:
            threeline[0] = "RETURN"
            threeline[1] = findword(listall[listype.index(thetype[-1])], disthreeline[1])[0]
            threeline[2] = findword(liststate, disthreeline[2])[0]
            if (threeline[2] == "BREAK"):
                insidefunction = False
                thetype.pop()
                listofindents[counter] = -1
    else:
        if (simpler[1] == 0):
            listall[simpler[2]].append(disthreeline[1])

            threeline[1] = disthreeline[1]
        else:
            threeline[1] = findword(listall[simpler[2]], disthreeline[1])[0]
        if (simpler[3] < len(listall)):
            threeline[2] = findword(listall[simpler[3]], disthreeline[2])[0]
        else:
            threeline[2] = disthreeline[2]
    if (threeline[0] == "WHILE"):
        listofindents[counter] = 1
        listofindents[int(threeline[2])] = -1
    elif (threeline[0] == "ITERATE"):
        listofindents[counter] = 1
        listofindents[int(threeline[2])] = -1
    elif ("COMPARE" in threeline[0] or threeline[0].startswith("IF") or threeline[0].endswith("IF")):
        listofindents[counter] = 1

        listofindents[int(threeline[2])] = -1
    elif (threeline[0] == "DEFINE"):
        listfunctions.append(threeline[0])
        listofindents[int(counter)] = 1
    counter += 1
    if (len(threeline[1]) == 0 or len(threeline[2]) == 0):
        problem = True
    return threeline
