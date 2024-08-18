def makeparenthasis(listofvals):
    stri = "("
    for i in range(len(listofvals) - 1):
        stri = stri + " " + str(listofvals[i]) + " " + ","
    stri = stri + " " + str(listofvals[-1]) + " )"
    return stri


def strreadvalue(type, name):
    return "getvar" + makeparenthasis([tostri(type), tostri(name)]) + ".read()"


lineupdate = "endline() ;"
infunction = False
dictoffunct = {i[0]: i for i in [[0]]}
dictofinstructions = {i: "thetext" for i in dictoffunct}
listfunctionswithtypes = {i[0]: i for i in [[0]]}
listfunctionswithtypes["GREATESTDIV"] = ["GREATESTDIV", "LIST", "LIST"]
for i in listfunctionswithtypes:
    for j in range(len(listfunctionswithtypes[i])):
        if (listfunctionswithtypes[i][j] == "BOOL"):
            listfunctionswithtypes[i][j] = "BOOLEAN"

listofindentchanges = [0 for i in range(1, 1000 + 1)]


def getinstructions(listfunctions, listezfunctions):
    global dictoffunct, listfunctionswithtypes
    dictoffunct = {i[0]: i for i in listezfunctions}
    listfunctionswithtypes = {i[0]: i for i in listfunctions}


def tostri(value):
    return "'" + str(value) + "'"


def MAKEINTEGER(name, value, linenum):
    global infunction
    inparan = makeparenthasis(['"INT"', tostri(name), value])
    if (infunction):
        declarestr = "addlocalvar" + inparan
    else:
        declarestr = "addvar" + inparan
    stri = "line(" + str(linenum) + ")" + "; " + declarestr + "; " + lineupdate
    return stri


def MAKESTR(name, value, linenum):
    global infunction
    inparan = makeparenthasis(['"STR"', tostri(name), "'" + str(value) + "'"])
    if (infunction):
        declarestr = "addlocalvar" + inparan
    else:
        declarestr = "addvar" + inparan
    stri = "line(" + str(linenum) + ")" + "; " + declarestr + "; " + lineupdate
    return stri


def MAKEBOOLEAN(name, value, linenum):
    global infunction
    inparan = makeparenthasis(['"BOOLEAN"', tostri(name), value])
    if (infunction):
        declarestr = "addlocalvar" + inparan
    else:
        declarestr = "addvar" + inparan
    stri = "line(" + str(linenum) + ")" + "; " + declarestr + "; " + lineupdate
    return stri


def NEWLIST(name, value, linenum):
    global infunction
    inparan = strreadvalue("INT", value)
    inparan = makeparenthasis(['"LIST"', tostri(name), inparan])

    if (infunction):
        declarestr = "addlocalvar" + inparan
    else:
        declarestr = "addvar" + inparan
    stri = "line(" + str(linenum) + ")" + "; " + declarestr + "; " + lineupdate
    return stri


def BASICCONDITION(name, compare, linenum):
    global infunction
    if (infunction == False):
        declarestr = "addcond" + makeparenthasis([tostri(name), tostri(compare)])
    else:
        declarestr = "addlocalcond" + makeparenthasis([tostri(name), tostri(compare)])

    stri = "line(" + str(linenum) + ")" + "; " + declarestr + "; " + lineupdate
    return stri


def LEFTSIDE(name, othername, linenum):
    thegetvar = "getvar" + makeparenthasis(['"INT"', tostri(othername)])
    stri = "line(" + str(linenum) + ")" + "; " + \
           "getcond" + makeparenthasis([tostri(name)]) + ".changeleft(" + thegetvar + ")" + "; " + lineupdate

    return (stri)


def RIGHTSIDE(name, othername, linenum):
    thegetvar = "getvar" + makeparenthasis(['"INT"', tostri(othername)])
    stri = "line(" + str(linenum) + ")" + "; " + \
           "getcond" + makeparenthasis([tostri(name)]) + ".changeright(" + thegetvar + ")" + "; " + lineupdate
    return (stri)


def CHANGECOMPARE(name, valuecompare, linenum):
    stri = "line(" + str(linenum) + ")" + "; " + \
           "getcond" + makeparenthasis([tostri(name)]) + ".changecompare(" + tostri(
        valuecompare) + ")" + "; " + lineupdate
    return (stri)


def WHILE(compare, endline, linenum):
    global listofindentchanges
    lineofwhile = "while" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getcond" + makeparenthasis([tostri(compare)])) + ".giveresult() and endline()"]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def ITERATE(listi, endline, linenum):
    global listofindentchanges
    lineofwhile = "for i in join" + makeparenthasis(["getvar('LIST'," + tostri(listi) + ")", str(linenum)]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return lineofwhile


def COMPARE(compare, endline, linenum):
    global listofindentchanges
    lineofwhile = "if" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getcond" + makeparenthasis([tostri(compare)])) + ".giveresult() and endline()"]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def ELSECOMPARE(compare, endline, linenum):
    global listofindentchanges
    lineofwhile = "elif" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getcond" + makeparenthasis([tostri(compare)])) + ".giveresult() and endline()"]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def WHILETRUE(bool, endline, linenum):
    global listofindentchanges
    lineofwhile = "while" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getvar('BOOLEAN'," + tostri(bool) + ").read() " + "and endline()")]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def IFTRUE(bool, endline, linenum):
    global listofindentchanges
    lineofwhile = "if" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getvar('BOOLEAN'," + tostri(bool) + ").read() " + "and endline()")]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def ELSEIF(bool, endline, linenum):
    global listofindentchanges
    lineofwhile = "elif" + makeparenthasis(["line(" + str(linenum) + ") and " + (
            "getvar('BOOLEAN'," + tostri(bool) + ").read() " + "and endline()")]) + ":"
    listofindentchanges[linenum + 1] = 1
    listofindentchanges[int(endline) + 1] = -1
    return (lineofwhile)


def INTEGERFUNCTION(name, type, linenum):
    global thetype, infunction
    infunction = True
    thetype = "INT"
    listofindentchanges[linenum + 1] = 1
    return "def " + name + "" + '():'


def STRINGFUNCTION(name, type, linenum):
    global thetype, infunction
    infunction = True
    thetype = "STR"
    listofindentchanges[linenum + 1] = 1
    return "def " + name + "" + '():'


def LISTFUNCTION(name, type, linenum):
    global thetype, infunction
    infunction = True
    thetype = "LIST"
    listofindentchanges[linenum + 1] = 1
    return "def " + name + "" + '():'


def RETURN(name, stay, linenum):
    if (stay == "BREAK"):
        listofindentchanges[linenum + 1] = -1
        global infunction
        infunction = False
    return ("line(" + str(linenum) + "); " + "return(updatelineexitingcall" + makeparenthasis(
        [tostri(thetype), tostri(name)]) + ")")


def PRINTSTRING(name, state, linenum):
    if (state == "BREAK"):
        state = "False"
    else:
        state = "True"
    return "line(" + str(linenum) + "); " + "Print(" + "getvar('STRING',name)," + state + "); " + "endline()"


def PRINTINTEGER(name, state, linenum):
    if (state == "BREAK"):
        state = "True"
    else:
        state = "False"
    return "line(" + str(linenum) + "); " + "Print(" + "getvar('INT'," + tostri(
        name) + ")," + state + "); " + "endline()"


def SETINDEX(name, index, linenum):
    return ("line(" + str(linenum) + "); getvar('LIST'," + tostri(name) + ").changeindex(" + strreadvalue("INT",
                                                                                                          index) + "); endline()")


def GETSTRING(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('STR'," + name + ").copyvar(getvar('LIST'," + listname + ").read()); endline()")


def GETINTEGER(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('INT'," + name + ").copyvar(getvar('LIST'," + listname + ").read()); endline()")


def GETLIST(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('LIST'," + name + ").copyvar(getvar('LIST'," + listname + ").read()); endline()")


def GETBOOL(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('BOOLEAN'," + name + ").copyvar(getvar('LIST'," + listname + ").read()); endline()")


def WRITESTRING(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(linenum) + ");getvar('LIST'," + listname + ") .placevalue(" + name + ',"STR"'"); endline()")


def WRITEINTEGER(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(linenum) + ");getvar('LIST'," + listname + ") .placevalue(" + name + ',"INT"'"); endline()")


def WRITEBOOL(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('LIST'," + listname + ") .placevalue(" + name + ',"BOOLEAN"'"); endline()")


def WRITELIST(listname, name, linenum):
    name = tostri(name)
    listname = tostri(listname)
    return ("line(" + str(linenum) + ");getvar('LIST'," + listname + ") .placevalue(" + name + ',"LIST"'"); endline()")


def GETTYPE(listname, strname, linenum):
    strname = tostri(strname)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('STR'," + strname + ").write(getvar('LIST'," + listname + ").returntype()); endline()")


def LENGTH(listname, intname, linenum):
    intname = tostri(intname)
    listname = tostri(listname)
    return ("line(" + str(
        linenum) + ");getvar('INT'," + intname + ").write(getvar('LIST'," + listname + ").getsize()); endline()")


def ADDVALUES(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "add" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def MULTIPLY(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "mult" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def MATHPOW(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "pow" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def DIVIDE(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "betterdiv" + makeparenthasis(
        [tostri(vali), tostri(vali2)]) + "; endline()")


def SIMPLEDIVIDE(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "div" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def SUBTRACT(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "dec" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def MODULO(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "mod" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def COMBINE(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "comb" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def ADDSIZE(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "addsize" + makeparenthasis([tostri(vali), tostri(vali2)]) + "; endline()")


def ASSSIGNINT(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "assignint" + makeparenthasis(
        [tostri(vali), tostri(vali2)]) + "; endline()")


def STRINGASSIGN(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "assignstr" + makeparenthasis(
        [tostri(vali), tostri(vali2)]) + "; endline()")


def COPYLIST(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "assignlist" + makeparenthasis(
        [tostri(vali), tostri(vali2)]) + "; endline()")


def BLANKSPACES(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "blankspaces" + makeparenthasis([tostri(vali), vali2]) + "; endline()")


def TYPETOINT(vali, vali2, linenum):
    return ("line(" + str(linenum) + "); " + "typetoint" + makeparenthasis(
        [tostri(vali), tostri(vali2)]) + "; endline()")


dictofinstructions["MAKEINTEGER"] = MAKEINTEGER
dictofinstructions["MAKESTR"] = MAKESTR
dictofinstructions["MAKEBOOLEAN"] = MAKEBOOLEAN
dictofinstructions["NEWLIST"] = NEWLIST
dictofinstructions["BASICCONDITION"] = BASICCONDITION
dictofinstructions["LEFTSIDE"] = LEFTSIDE
dictofinstructions["RIGHTSIDE"] = RIGHTSIDE
dictofinstructions["CHANGECOMPARE"] = CHANGECOMPARE
dictofinstructions["WHILE"] = WHILE
dictofinstructions["ITERATE"] = ITERATE
dictofinstructions["COMPARE"] = COMPARE
dictofinstructions["ELSECOMPARE"] = ELSECOMPARE
dictofinstructions["WHILETRUE"] = WHILETRUE
dictofinstructions["IFTRUE"] = IFTRUE
dictofinstructions["ELSEIF"] = ELSEIF
dictofinstructions["SETINDEX"] = SETINDEX
dictofinstructions["INTEGERFUNCTION"] = INTEGERFUNCTION
dictofinstructions["STRINGFUNCTION"] = STRINGFUNCTION
dictofinstructions["LISTFUNCTION"] = LISTFUNCTION
dictofinstructions["PRINTSTRING"] = PRINTSTRING
dictofinstructions["PRINTINTEGER"] = PRINTINTEGER
dictofinstructions["GETSTRING"] = GETSTRING
dictofinstructions["GETINTEGER"] = GETINTEGER
dictofinstructions["GETLIST"] = GETLIST
dictofinstructions["GETBOOL"] = GETBOOL
dictofinstructions["WRITESTRING"] = WRITESTRING
dictofinstructions["WRITEINTEGER"] = WRITEINTEGER
dictofinstructions["WRITEBOOL"] = WRITEBOOL
dictofinstructions["WRITELIST"] = WRITELIST
dictofinstructions["GETTYPE"] = GETTYPE
dictofinstructions["LENGTH"] = LENGTH
dictofinstructions["ASSSIGNINT"] = ASSSIGNINT
dictofinstructions["ADDSIZE"] = ADDSIZE
dictofinstructions["STRINGASSIGN"] = STRINGASSIGN
dictofinstructions["COPYLIST"] = COPYLIST
dictofinstructions["ADDVALUES"] = ADDVALUES
dictofinstructions["MULTIPLY"] = MULTIPLY
dictofinstructions["MATHPOW"] = MATHPOW
dictofinstructions["DIVIDE"] = DIVIDE
dictofinstructions["SIMPLEDIVIDE"] = COPYLIST
dictofinstructions["SUBTRACT"] = SUBTRACT
dictofinstructions["MODULO"] = MODULO
dictofinstructions["COMBINE"] = COMBINE
dictofinstructions["BLANKSPACES"] = BLANKSPACES
dictofinstructions["ADDSIZE"] = ADDSIZE
dictofinstructions["COPYLIST"] = COPYLIST
dictofinstructions["RETURN"] = RETURN
dictofinstructions["TYPETOINT"] = TYPETOINT

listoflists = ""


def makepredict(listi, i):
    if (listi[0] in dictofinstructions):
        return dictofinstructions[listi[0]](listi[1], listi[2], i)
    else:
        listfun = listfunctionswithtypes[listi[0]]
        return ("updatelinewithcall" + makeparenthasis(
            [tostri(listfun[1]), tostri(listi[1]), listi[0], tostri(listfun[2]), tostri(listi[2]), i]))


def makepyfile(listi):
    f = open(r"test.py", 'w+')
    f.write("from createdpython import * \n")
    counterindent = 0
    indent = "    "
    for i in range(1, len(listi) + 1):
        counterindent += listofindentchanges[i]
        f.write(indent * counterindent + makepredict(listi[i - 1], i) + '\n')
    f.write("printvars()")


if __name__ == '__main__':
    listi = [["MAKEINTEGER", "THEINT", '2769'], ["MAKEINTEGER", "THEINTI", '1065'], ["MAKEINTEGER", "THROWONE", '1065'],
             ["MAKEINTEGER", "THROWTWO", '1065'], ["NEWLIST", "LISTOFTWO", '2'], ["SETINDEX", "LISTOFTWO", '0'],
             ["WRITEINTEGER", "LISTOFTWO", 'THEINT'], ["SETINDEX", "LISTOFTWO", '1'],
             ["WRITEINTEGER", "LISTOFTWO", 'THEINTI'], ["MAKEINTEGER", "ZERO", '0'], ["ADDVALUES", "THEINT", 'THEINTI'],
             ["PRINTINTEGER", "TEMPORARY", 'BREAK'], ["LISTFUNCTION", "GREATESTDIV", 'LIST'],
             ["SETINDEX", "LISTOFTWO", '0'], ["GETINTEGER", "LISTOFTWO", 'THROWONE'], ["SETINDEX", "LISTOFTWO", '1'],
             ["GETINTEGER", "LISTOFTWO", 'THROWTWO'], ["BASICCONDITION", "EUCLIDCOMPARE", 'EQUALS'],
             ["LEFTSIDE", "EUCLIDCOMPARE", 'THROWTWO'], ["RIGHTSIDE", "EUCLIDCOMPARE", 'ZERO'],
             ["COMPARE", "EUCLIDCOMPARE", '23'], ["WRITEINTEGER", "LISTOFTWO", 'THROWTWO'],
             ["RETURN", "LISTOFTWO", "STAY"], ["RIGHTSIDE", "EUCLIDCOMPARE", 'THROWTWO']
        , ["SETINDEX", "LISTOFTWO", '0'], ["WRITEINTEGER", "LISTOFTWO", 'THROWTWO'], ["MODULO", "THROWONE", 'THROWTWO'],
             ["SETINDEX", "LISTOFTWO", '1'], ["WRITEINTEGER", "LISTOFTWO", 'TEMPORARY'],
             ["GREATESTDIV", "LISTOFTWO", 'LISTOFTWO'], ["RETURN", "LISTOFTWO", 'BREAK'],
             ["GREATESTDIV", "LISTOFTWO", 'LISTOFTWO']]
    makepyfile(listi)
