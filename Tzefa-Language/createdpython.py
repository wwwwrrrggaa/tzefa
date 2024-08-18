### finally found the bug
### here it is:
### lets say function f calls a new instance of itself
### the new instance creates cond b and a new instance of itself
### the new instance closes and returns to the instance with  cond b that also closes itself
### now the original instance still owns cond b since no function was called to clean the stack or dict between the exiting calls
dicte = {"EQUALS": 0, "BIGEQUALS": 1, "BIGGER": 2}
listfunctions = [(lambda x, y: x == y), (lambda x, y: x >= y), (lambda x, y: x > y)]
import sys
import Number2Name
linecount = 0
currentline = 0
linelimit = 1000
functionlimit = 25
functioncount = 0
printed = ""
alltheconds = {}
reserveconds = {}
reserveloc = {}
reserveglob = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}


def line(linenum):
    global currentline
    currentline = linenum
    # if("LISTOFTWO" in allthevars["LIST"]):
    # print(linenum,getvar("LIST","LISTOFTWO").values)
    return True


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def setnext(self, nextvalue):
        self.next = Node(value=nextvalue)

    def setnextnode(self, next):
        self.next = next

    def getnext(self):
        return self.next

    def getvalue(self):
        return self.value


class Stack:
    def __init__(self):
        self.top = None
        self.list = []

    def isempty(self):
        return len(self.list) == 0

    def push(self, value):
        self.list.append(value)

    def pop(self):
        return self.list.pop()


functioncalls = Stack()


def addcond(name, compare):
    global alltheconds
    if (name in alltheconds):
        errore.varexistserror(name)
    else:
        alltheconds[name] = COND(compare)


def addlocalcond(name, compare):
    global alltheconds, allthelocalconds
    if (name in alltheconds):
        errore.varexistserror(name)
    else:
        alltheconds[name] = COND(compare)
        allthelocalconds[name] = alltheconds[name]
    return allthelocalconds


# import dis
# print(dis.dis(addlocalcond))
def movetonewconds(localconds):
    global stackoflocalconds
    global alltheconds
    global reserveconds
    for i in localconds:
        del alltheconds[i]
    stackoflocalconds.push(localconds)


def backtooldconds():
    global stackoflocalconds
    popped = stackoflocalconds.pop()
    global alltheconds
    global reserveconds
    for i in reserveconds:
        if i in alltheconds:
            del alltheconds[i]
    reserveconds = {}
    for i in popped:
        alltheconds[i] = popped[i]
        reserveconds[i] = popped[i]
    return popped


def getcond(name):
    global alltheconds
    if (name in alltheconds):
        return alltheconds[name]
    else:
        errore.doesntexisterror(name)


def printvars():
    global allthevars, printed
    print("END OF PROGRAM")
    print()
    for i in allthevars:
        print("All the vars used from type " + i)
        for j in allthevars[i]:
            if (allthevars[i][j].iswritable()):
                print(j + " : " + allthevars[i][j].tostring())
            else:
                pass
        print("")
    print("All that was printed during the program")
    print(printed)


def addvar(type, name, value):
    global allthevars
    if (name in allthevars[type]):
        errore.varexistserror(name)
    else:
        pass
    if (type == "LIST"):
        allthevars[type][name] = LIST(name, value, True, True, type)
    else:
        allthevars[type][name] = VALUE(name, value, True, True, type)


def getvar(type, name):
    global allthevars
    if (name in allthevars[type]):
        return allthevars[type][name]
    else:
        errore.doesntexisterror(name)


def addlocalvar(type, name, value):
    global dictlocalvars, allthevars
    allthelocalrvars = dictlocalvars
    if (name in allthevars[type]):
        errore.varexistserror(name)
    else:
        pass
    if (type == "LIST"):
        allthevars[type][name] = LIST(name, value, True, True, type)
        allthelocalrvars[type][name] = allthevars[type][name]

    else:
        allthevars[type][name] = VALUE(name, value, True, True, type)
        allthelocalrvars[type][name] = allthevars[type][name]


def switchtonewcall(dict1: dict, dict2: dict, stack: Stack):
    for i in dict2:
        for j in dict2[i]:
            del dict1[i][j]
    stack.push(dict2)
    return {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}


def returntoolastcall(dict1: dict):
    stack = localsvars
    lastcall = stack.pop()
    global reserveloc
    for i in reserveloc:
        for j in reserveloc[i]:
            if j in dict1[i]:
                del dict1[i][j]
    reserveloc = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}
    for i in lastcall:
        for j in lastcall[i]:
            dict1[i][j] = lastcall[i][j]
            reserveloc[i][j] = lastcall[i][j]
    return lastcall


def Print(var, newline):
    global printed
    toprint = var.tostring() + newline * '\n' + ' ' * (1 - newline)
    print(toprint, end='')
    printed = printed + toprint


class VALUE:
    def __init__(self, name, value, readable, writable, TYPE):
        self.name = name
        self.value = value
        self.readable = readable
        self.writable = writable
        self.type = TYPE

    def write(self, value):
        if (self.writable == True):
            self.value = value
        else:
            errore.writeerror(self.name, value)

    def forcewrite(self, value):
        self.value = value

    def read(self):
        if (self.readable == True):
            return self.value
        else:
            errore.readerror(self.name)

    def forceread(self):
        return self.value

    def changeread(self, readstatus):
        self.readable = readstatus

    def changewrite(self, writestatus):
        self.writable = writestatus

    def getname(self):
        return self.name

    def iswritable(self):
        return self.writable

    def isreadable(self):
        return self.readable

    def tostring(self):
        return str(self.value)

    def givetype(self):
        return self.type

    def override(self, value):
        self.value = value

    def makecopy(self):
        return VALUE(self.name, self.value, self.readable, True, self.type)

    def copyvar(self, vari):
        if (self.type != vari.type):
            errore.typeerror(self.name, self.type, vari.type)
        else:
            if (vari.isreadable() == False):
                errore.readerror(vari.getname())
            else:
                self.value = vari.value


def add(Vali1, vali2):
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() + getvar("INT", vali2).read())


def dec(Vali1, vali2):
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() - getvar("INT", vali2).read())

def mult(Vali1, vali2):
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() * getvar("INT", vali2).read())


def div(Vali1, vali2):
    if (getvar("INT", vali2).read() == 0):
        errore.DIVZEROERROR(vali2)
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() // getvar("INT", vali2).read())


def betterdiv(Vali1, vali2):
    if (getvar("INT", vali2).read() == 0):
        errore.DIVZEROERROR(vali2)
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() / getvar("INT", vali2).read())


def pow(Vali1, vali2):
    getvar("INT", "TEMPORARY").forcewrite(getvar("INT", Vali1).read() ** getvar("INT", vali2).read())


def mod(Vali1, vali2):
    if (getvar("INT", vali2).read() == 0):
        errore.DIVZEROERROR(vali2)
    getvar("INT", "TEMPORARY").forcewrite(int(getvar("INT", Vali1).read() % getvar("INT", vali2).read()))


def comb(Vali1, vali2):
    getvar("STR", "TEMPSTRING").forcewrite(getvar("STR", Vali1).read() + getvar("STR", vali2).read())


def addsize(Vali1, vali2):
    getvar("LIST", Vali1).addsize(getvar("INT", vali2).read())


def assignlist(Vali1, vali2):
    getvar("LIST", Vali1).copyvar(getvar("LIST", vali2))


def assignstr(Vali1, vali2):
    getvar("STR", Vali1).copyvar(getvar("STR", vali2))


def assignint(Vali1, vali2):
    getvar("INT", Vali1).copyvar(getvar("INT", vali2))


def blankspaces(Vali1, Vali2):
    getvar("STR", Vali1).write(getvar("STR", Vali1).read() + ' ' * Vali2)


def typetoint(vali1, vali2):
    lookuptable = {"INT": 0, "STR": 1, "BOOLEAN": 2, "LIST": 3}
    if (getvar('STR', vali1).read() in lookuptable):
        getvar('INT', vali2).write(lookuptable[getvar('STR', vali1).read()])
    else:
        errore.typetointerror(getvar('STR', vali1).read())


class COND:
    def __init__(self, compare):
        self.index = dicte[compare]
        self.left = VALUE("0", 0, False, False, "INT")
        self.right = VALUE("0", 0, False, False, "INT")

    def changecompare(self, compare):
        self.index = dicte[compare]

    def changeleft(self, left):
        self.left = left

    def changeright(self, right):
        self.right = right

    def giveresult(self):
        return listfunctions[self.index](self.left.read(), self.right.read())

    def givetype(self):
        return self.type


class EERROR(Exception):
    global sizeoflistinuse, wantedindex

    def __init__(self):
        pass

    def nameerror(self, type, name):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Var of name " + name + " doesn't exist as type " + type)
        print(" ")
        printvars()
        sys.exit(0)

    def makeeindexrror(self, sizeoflistinuse, wantedindex, name, name2):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Tried to change index of list " + name + " with size of " + str(sizeoflistinuse) + " to value " + str(
            wantedindex) + " placed in " + str(name2))
        print(" ")
        printvars()
        sys.exit(0)

    def DIVZEROERROR(self, name):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Cant divide by zero and var " + name + " has value of zero")
        print(" ")
        printvars()
        sys.exit(0)

    def doesntexisterror(self, name):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("No object with name " + str(name) + " exists")
        printvars()
        sys.exit(0)

    def writeerror(self, name, value):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Tried to write value of " + str(value) + " to unwritable variable " + name)
        print(" ")
        printvars()
        sys.exit(0)

    def typetointerror(self, value):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("No type such as " + str(value))
        print(" ")
        printvars()
        sys.exit(0)

    def readerror(self, name):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Tried to read from unreadable variable " + name)
        print(" ")
        printvars()
        sys.exit(0)

    def linelimiterror(self):
        global currentline
        globalline = currentline
        print("Error Line: " + str(currentline))
        print("Program ran for too long")
        print(" ")
        printvars()
        sys.exit(0)

    def overflowerror(self, functioncalls):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Executing too many  function calls ")
        print("List of function calls")
        for i in functioncalls:
            pass
        print(" ")
        printvars()
        sys.exit(0)

    def cantchangeindexerror(self, name, value):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Tried to change indexes of list " + name + " and add size  " + str(value) + " but list is unwritable")

    def varexistserror(self, name):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Tried to create object with name " + name + " but var already exists")
        print(" ")
        printvars()
        sys.exit(0)

    def typeerror(self, name, type1, type2):
        global currentline
        globalline = currentline
        print("Error Line: " + str(globalline))
        print("Mismatch of types  " + type1 + " and " + type2 + " in list " + name)
        print(" ")
        printvars()
        sys.exit(0)


class LIST:
    def __init__(self, name, size, readable, writable, TYPE):
        self.size = size
        self.currentindex = 0
        self.values = [VALUE(name=(str(name) + " " + str(i)), value=0, writable=True, readable=True, TYPE="INT") for i
                       in range(size)]
        self.types = ["INT" for i in range(size)]
        self.readable = readable
        self.writable = writable
        self.name = name
        self.type = TYPE

    def addsize(self, added):
        if (self.writable):
            self.size = self.size + added
            self.values = [
                self.values[i] if listfunctions[2](self.size, i) else VALUE(name=(str(i) + " " + str(i)), value=0,
                                                                            writable=True, readable=True, TYPE="INT")
                for i in range(self.size + added)]
        else:
            errore.cantchangeindexerror(self.name, added)

    def forceaddsize(self, added):
        self.size = self.size + added
        self.values = [
            self.values[i] if listfunctions[2](self.size, i) else VALUE(name=(str(i) + " " + str(i)), value=0,
                                                                        writable=True, readable=True, TYPE="INT") for i
            in range(self.size + added)]

    def changeindex(self, newindex):
        if (self.readable):
            if (newindex >= self.size):
                errore.makeeindexrror(newindex, self.size, self.name)
            else:
                self.index = newindex
        else:
            errore.readerror(self.name)

    def forcechangeindex(self, newindex):
        if (newindex >= self.size):
            errore.makeeindexrror(newindex, self.size, self.name)
        else:
            self.index = newindex

    def placevalue(self, name, type):
        if (self.writable):
            thevar = getvar(type, name)
            if (thevar.isreadable()):
                self.types[self.index] = type
                self.values[self.index] = thevar.makecopy()
            else:
                errore.readerror(name)
        else:
            errore.writeerror(self.name, name)

    def forceplacevalue(self, name, type):
        thevar = getvar(type, name)
        if (thevar.isreadable()):
            self.types[self.index] = type
            self.values[self.index] = thevar.makecopy()
        else:
            errore.readerror()

    def returnvalue(self):
        if (self.readable):
            return self.values[self.index]
        else:
            errore.readerror(self.name)

    def read(self):
        if (self.readable):
            return self.values[self.index]
        else:
            errore.readerror(self.name)

    def forcereturnvalue(self):
        return self.values[self.index]

    def copybyvalue(self, newvalue: VALUE):
        if (self.types[self.index] == newvalue.givetype()):
            newvalue.write(self.values[self.index].read())
        else:
            errore.typeerror(name=self.name, type1=self.types[self.index], type2=newvalue.givetype())

    def returntype(self):
        if (self.readable):
            return self.types[self.index]
        else:
            errore.readerror(self.name)

    def forcereturntype(self):
        return self.types[self.index]

    def tostring(self):
        strei = ""
        for i in self.values:
            strei = strei + str(i.tostring()) + " "
        return "[ " + strei + " ]"

    def tostringoftypes(self):
        if (self.readable):
            stre = ""
            for i in self.types:
                stre = stre + i[0]
            return stre
        else:
            errore.readerror(self.name)

    def forcetostringoftypes(self):
        stre = ""
        for i in self.types:
            stre = stre + i[0]
        return stre

    def changeread(self, readstatus):
        self.readable = readstatus

    def changewrite(self, writestatus):
        self.writable = writestatus

    def getname(self):
        return self.name

    def iswritable(self):
        return self.writable

    def isreadable(self):
        return self.readable

    def getvalues(self):
        return self.values

    def gettypes(self):
        return self.types

    def getsize(self):
        return self.size

    def makecopy(self):
        thelist = LIST(self.name, self.size, self.readable, True, self.type)
        thelist.types = self.types.copy()
        thelist.values = [val.makecopy() for val in self.values]
        return thelist

    def override(self, values, types, size):
        self.values = values
        self.types = types
        self.size = size

    def givetype(self):
        return self.type

    def copyvar(self, listi):
        if (self.type != listi.type):
            errore.typeerror(self.name, self.type, listi.type)
        else:
            if (listi.isreadable() == False):
                errore.readerror(listi.getname())
            else:
                self.type = 'LIST'
                self.types = listi.types.copy()
                self.size = listi.size
                self.values = [var.makecopy() for var in listi.values]


def join(listi: LIST, linee):
    if (listi.isreadable() == False):
        errore.readerror(listi.name)
    for i in range(listi.size):
        listi.index = i
        value = listi.values[i]
        typi = listi.types[i]
        goodloopvars[typi].copyvar(value)
        getvar('STR', 'TEMPSTRING').forcewrite(typi)
        line(linee)
        endline()
        yield value


def returntooldlocals(dictofglobals, dictoflocals):
    global reserveglob
    for i in reserveglob:
        for j in reserveglob[i]:
            del dictofglobals[i][j]
    reserveglob = {"INT": {}, "STR": {}, "LIST": {}}
    for i in dictoflocals:
        for j in dictoflocals[i]:
            dictofglobals[i][j] = dictoflocals[i][j]
            reserveglob[i][j] = dictoflocals[i][j]


def endline():
    global errore, linecount
    linecount += 1
    if (linecount == linelimit):
        errore.linelimiterror()
    else:
        return True


def updateline():
    global errore, linecount
    linecount += 1
    if (linecount == linelimit):
        errore.linelimiterror()
    else:
        return True


def updatelinewithcall(type, namevarinput, function, typeoutput, nameoutput, lini):
    line(lini)
    global allthevars, functioncount, localsstack, programlocals, nameofprogramlocals, dictlocalvars, functionlimit, allthelocalconds, nameofprogramlocals
    localsstack.push(programlocals)

    varinput = getvar(type, namevarinput)
    for i in nameofprogramlocals:
        if (i == 'STR'):
            allthevars[i][nameofprogramlocals[i]] = VALUE(nameofprogramlocals[i], '', False, False, 'STR')
        elif (i == 'INT'):
            allthevars[i][nameofprogramlocals[i]] = VALUE(nameofprogramlocals[i], 0, False, False, 'INT')
        else:
            allthevars[i][nameofprogramlocals[i]] = LIST(nameofprogramlocals[i], 8, False, False, 'LIST')
    programlocals = {"INT": {"LOCALINT": allthevars["INT"]["LOCALINT"]},
                     "STR": {"LOCALSTR": allthevars["STR"]["LOCALSTR"]},
                     "LIST": {"LOCALLIST": allthevars["LIST"]["LOCALLIST"]}}
    allthevars[type][nameofprogramlocals[type]].copyvar(varinput)
    vartosend = allthevars[type][nameofprogramlocals[type]]
    functioncount += 1
    if (functioncount == functionlimit):
        errore.overflowerror(functioncalls)
    else:
        dictlocalvars = switchtonewcall(allthevars, dictlocalvars, localsvars)
        vartosend.changeread(True)
        vartosend.changewrite(True)
        movetonewconds(allthelocalconds)
        allthelocalconds = {}
        output = getvar(typeoutput, nameoutput)
        outi = function()
        output.copyvar(outi)
        endline()


def updatelineexitingcall(type, namevaroutput):
    global allthevars, localsstack, functioncount, allthelocalconds, dictlocalvars, localsvars
    thevar = getvar(type, namevaroutput)
    functioncount = functioncount - 1
    popped = localsstack.pop()
    ### because i can't remember
    ### get old "global"
    returntooldlocals(allthevars, popped)
    ### get old locals
    dictlocalvars = returntoolastcall(allthevars)
    ### get old conds
    allthelocalconds = backtooldconds()
    endline()
    return thevar

localsvars = Stack()  #### the stack for locals created in function
localsvars.push({"INT": {}, "STR": {}, "LIST": {}})
localsstack = Stack()  ###### the stack of the program locals
LOOPINTEGER = VALUE(name="LOOPINTEGER", value=0, readable=True, writable=False, TYPE="INT")
LOOPSTRING = VALUE(name="LOOPSTRING", value="", readable=True, writable=False, TYPE="STR")
LOOPBOOL = VALUE(name="LOOPBOOL", value=True, readable=True, writable=False, TYPE="BOOL")
LOOPLIST = LIST(name="LOOPLIST", size=8, readable=True, writable=False, TYPE="LIST")
TEMPORARY = VALUE(name="TEMPORARY", value=0, readable=True, writable=True, TYPE="INT")
LOCALINT = VALUE(name="LOCALINT", value=0, readable=0, writable=0, TYPE="INT")
loopvars = {"INT": {"LOOPINTEGER": LOOPINTEGER}, "STR": {"LOOPSTRING": LOOPSTRING}, "LIST": {"LOOPLIST": LOOPLIST},
            "BOOLEAN": {"LOOPBOOL": LOOPBOOL}}
goodloopvars = {"INT": LOOPINTEGER, "STR": LOOPSTRING, "LIST": LOOPLIST, "BOOLEAN": LOOPBOOL}
TEMPSTRING = VALUE(name="TEMPSTRING", value="", readable=True, writable=False, TYPE="STR")
LOCALSTR = VALUE(name="LOCALSTR", value="", readable=0, writable=0, TYPE="STR")
INTEGER = VALUE(name="INTEGER", value="INT", readable=True, writable=False, TYPE="STR")
STRING = VALUE(name="STRING", value="STR", readable=True, writable=False, TYPE="STR")
LISTI = VALUE(name="LIST", value="LIST", readable=True, writable=False, TYPE="STR")
BOOLEAN = VALUE(name="BOOLEAN", value="BOOLEAN", readable=True, writable=False, TYPE="STR")
LOCALLIST = LIST(name="LOCALLIST", size=8, readable=0, writable=0, TYPE="LIST")
THETRUTH = COND('EQUALS')
THETRUTH.changeleft(TEMPORARY)
THETRUTH.changeright(TEMPORARY)
allthevars = {"INT": {"LOOPINTEGER": LOOPINTEGER, "TEMPORARY": TEMPORARY, "LOCALINT": LOCALINT},
              "STR": {"LOOPSTRING": LOOPSTRING, "TEMPSTRING": TEMPSTRING, "LOCALSTR": LOCALSTR, "INTEGER": INTEGER,
                      "STRING": STRING, "LIST": LISTI, "BOOLEAN": BOOLEAN},
              "LIST": {"LOOPLIST": LOOPLIST, "LOCALLIST": LOCALLIST}, "BOOLEAN": {"LOOPBOOL": LOOPBOOL}}
for i in range(101):
   allthevars["INT"][Number2Name.get_name(i)]=VALUE(name=Number2Name.get_name(i),value=i,readable=True, writable=False, TYPE="INT")
examplelocalvars = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}
dictlocalvars = examplelocalvars.copy()
programlocals = {"INT": {"LOCALINT": allthevars["INT"]["LOCALINT"]}, "STR": {"LOCALSTR": allthevars["STR"]["LOCALSTR"]},
                 "LIST": {"LOCALLIST": allthevars["LIST"]["LOCALLIST"]}}
nameofprogramlocals = {"INT": "LOCALINT", "STR": "LOCALSTR", "LIST": "LOCALLIST"}
stackoflocalconds = Stack()
localsstack.push(programlocals)
dictlocalvars = examplelocalvars.copy()
allthelocalconds = {}
errore = EERROR()
alltheconds['THETRUTH'] = THETRUTH
