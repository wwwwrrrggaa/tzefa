from Tzefa_Language import Number2Name
from fast_edit_distance import edit_distance


def giveinstructions():
    ### returns instructions for each function in the language for topy
    return listfunctions, listezfunc


listofindents = []


def updatesizelistofindnets(size):
    global listofindents
    listofindents = [0] * (size + 1)


def tosimple(func):
    simpler = ["a", "b", "c", "d"]
    simpler[0] = func[0]
    if func[1].startswith("NEW"):
        simpler[1] = 0
    else:
        simpler[1] = 1
    i = 1
    j = 2
    if func[i].endswith("INT"):
        simpler[j] = 0
    elif func[i].endswith("STR"):
        simpler[j] = 1
    elif func[i].endswith("LIST"):
        simpler[j] = 2
    elif func[i].endswith("BOOL"):
        simpler[j] = 3
    elif func[i].endswith("COND"):
        simpler[j] = 4
    elif func[i].endswith("STATE"):
        simpler[j] = 5
    elif func[i].endswith("TYPE"):
        simpler[j] = 6
    elif func[i].endswith("FUNC"):
        simpler[j] = 7
    elif func[i].endswith("TRUTH"):
        simpler[j] = 8
    elif func[i].endswith("COMPARE"):
        simpler[j] = 9
    elif func[i].endswith("NUMNAME"):
        simpler[j] = 10
    elif func[i].endswith("TEXT"):
        simpler[j] = 11
    i = 2
    j = 3
    if func[i].endswith("INT"):
        simpler[j] = 0
    elif func[i].endswith("STR"):
        simpler[j] = 1
    elif func[i].endswith("LIST"):
        simpler[j] = 2
    elif func[i].endswith("BOOL"):
        simpler[j] = 3
    elif func[i].endswith("COND"):
        simpler[j] = 4
    elif func[i].endswith("STATE"):
        simpler[j] = 5
    elif func[i].endswith("TYPE"):
        simpler[j] = 6
    elif func[i].endswith("FUNC"):
        simpler[j] = 7
    elif func[i].endswith("TRUTH"):
        simpler[j] = 8
    elif func[i].endswith("COMPARE"):
        simpler[j] = 9
    elif func[i].endswith("NUMNAME"):
        simpler[j] = 10
    elif func[i].endswith("TEXT"):
        simpler[j] = 11
    simpler.append(0)
    return simpler



# CHANGED: Replaced NUM/INT with NUMNAME for immediate value reading (Index 10)
listfunctions = [
    ["MAKEINTEGER", "NEWINT", "NUMNAME"],
    ["MAKEBOOLEAN", "NEWBOOL", "TRUTH"],
    ["MAKESTR", "NEWSTR", "TEXT"],
    ["NEWLIST", "NEWLIST", "NUMNAME"],
    ["BASICCONDITION", "NEWCOND", "COMPARE"],
    ["LEFTSIDE", "COND", "INT"],
    ["RIGHTSIDE", "COND", "INT"],
    ["CHANGECOMPARE", "COND", "COMPARE"],
    ["WHILE", "COND", "NUMNAME"],
    ["COMPARE", "COND", "NUMNAME"],
    ["ELSECOMPARE", "COND", "NUMNAME"],
    ["ITERATE", "LIST", "NUMNAME"],
    ["WHILETRUE", "BOOL", "NUMNAME"],
    ["IFTRUE", "BOOL", "NUMNAME"],
    ["ELSEIF", "BOOL", "NUMNAME"],
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
    ["BLANKSPACES", "STR", "NUMNAME"],
    ["ADDSIZE", "LIST", "INT"],
    ["ASSSIGNINT", "INT", "INT"],
    ["STRINGASSIGN", "STR", "STR"],
    ["COPYLIST", "LIST", "LIST"],
]
listsimplefunc = [tosimple(i) for i in listfunctions]
listofindents = []
listezfunc = [i[0] for i in listfunctions]

# Variable Names only
listintegers = ["TEMPORARY", "LOCALINT", "LOOPINTEGER"]

# Immediate Number Names (Index 10)
listnumnames = []
word_to_num = {}
for i in range(101):
    name = Number2Name.get_name(i)
    listnumnames.append(name)
    word_to_num[name] = str(i)

liststrings = ["TEMPSTRING", "GLOBALSTR", "LOOPSTRING", "INTEGER", "STRING", "LIST", "BOOLEAN"]
listlists = ["GLOBALLIST", "LOOPLIST"]
listconds = ["THETRUTH"]
listbools = ["LOOPBOOL"]
liststate = ["STAY", "BREAK"]
listype = ["INTEGER", "STRING", "LIST", "BOOLEAN"]
lookuptype = {"INTEGER": "INT", "STRING": "STR", "LIST": "LIST", "BOOLEAN": "BOOL"}
listtruth = ["TRUE", "FALSE"]
listcompare = ["EQUALS", "BIGEQUALS", "BIGGER"]
listtext = []  # Placeholder for raw text arguments (Index 11)

listall = [
    listintegers,
    liststrings,
    listlists,
    listbools,
    listconds,
    liststate,
    listype,
    listezfunc,
    listtruth,
    listcompare,
    listnumnames,  # Index 10: Immediates
    listtext,  # Index 11: Text
]
thetype = []
insidefunction = False
counter = 0


def getsimples():
    return listsimplefunc


def sendlines(i):
    global listofindents
    listofindents = [0] * max(i + 1, 1000)


def giveindents():
    return listofindents

def ocr_edit_distance(word1, word2):
    """
    Custom Levenshtein distance tailored for OCR.
    Heavily penalizes distant letter swaps (H vs Q), but forgives common OCR shapes.
    """
    word1, word2 = word1.upper(), word2.upper()

    # Common OCR confusions get a low penalty (0.5).
    # Add more to this dictionary as you find specific model confusions!
    low_cost_subs = {
        ('O', '0'): 0.5, ('0', 'O'): 0.5,
        ('I', '1'): 0.5, ('1', 'I'): 0.5,
        ('I', 'L'): 0.5, ('L', 'I'): 0.5,
        ('S', '5'): 0.5, ('5', 'S'): 0.5,
        ('Z', '2'): 0.5, ('2', 'Z'): 0.5,
        ('C', 'O'): 0.5, ('O', 'C'): 0.5,
        ('C', 'G'): 0.5, ('G', 'C'): 0.5,
        ('B', '8'): 0.5, ('8', 'B'): 0.5,
        ('D', 'O'): 0.5, ('O', 'D'): 0.5,
        ('E', 'F'): 0.5, ('F', 'E'): 0.5,
        ('A', '4'): 0.5, ('4', 'A'): 0.5,
    }

    m, n = len(word1), len(word2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * 1.0  # Cost of deletion
    for j in range(n + 1):
        dp[0][j] = j * 1.0  # Cost of insertion

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                cost = 0.0
            else:
                sub_pair = (word1[i-1], word2[j-1])
                # Generic substitution is penalized heavily (2.0)
                cost = low_cost_subs.get(sub_pair, 2.0)

            dp[i][j] = min(
                dp[i-1][j] + 1.0,      # deletion
                dp[i][j-1] + 1.0,      # insertion
                dp[i-1][j-1] + cost    # substitution
            )
    return dp[m][n]

def findword(somelist, word):
    min_dist = 999
    tobereturned = [word, 0]
    lentobereturned = 16
    for b in range(len(somelist)):
        lenword = len(word)
        i = somelist[b]
        lenofi = len(i)
        if i == word:
            return [i, b]
        else:
            distance = edit_distance(word, i, 4)
            if distance < min_dist:
                min_dist = distance
                tobereturned = [i, b]
                lentobereturned = len(tobereturned[0])
            elif distance == min_dist:
                if abs(lenword - lenofi) < abs(lenword - lentobereturned):
                    tobereturned = [i, b]
                    lentobereturned = len(tobereturned[0])

    return tobereturned


def handelfirstword(firstword):
    func, index = findword(listezfunc, firstword)
    # Check if Arg2 (Index 2 in definition) is NUMNAME (Index 10 in listall)
    # We use listfunctions directly to check the string type
    if listfunctions[index][2] == "NUMNAME":
        # Return 1 to indicate number/immediate processing
        return (func, index, 1)
    else:
        return (func, index, 0)


def toline(line, index, listofindents):
    global counter
    global thetype
    global insidefunction
    disthreeline = line.split(" ")
    threeline = ["", "", ""]
    problem = False
    threeline[0] = listezfunc[index]
    simpler = listsimplefunc[index]
    if threeline[0] == "STRINGFUNCTION":
        if insidefunction:
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
    elif threeline[0] == "INTEGERFUNCTION":
        if insidefunction:
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
    elif threeline[0] == "LISTFUNCTION":
        if insidefunction:
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
    elif simpler[0] == "RETURN":
        if len(thetype) == 0:
            problem = True
        else:
            threeline[0] = "RETURN"
            threeline[1] = findword(listall[listype.index(thetype[-1])], disthreeline[1])[0]
            threeline[2] = findword(liststate, disthreeline[2])[0]
            if threeline[2] == "BREAK":
                insidefunction = False
                thetype.pop()
                listofindents[counter] = -1
    else:
        # Arg 1
        if simpler[1] == 0:
            listall[simpler[2]].append(disthreeline[1])
            threeline[1] = disthreeline[1]
        else:
            threeline[1] = findword(listall[simpler[2]], disthreeline[1])[0]
            # If Arg 1 is a NUMNAME, replace with actual value
            if simpler[2] == 10:
                threeline[1] = word_to_num[threeline[1]]

        # Arg 2
        if simpler[3] < len(listall):
            threeline[2] = findword(listall[simpler[3]], disthreeline[2])[0]
            # If Arg 2 is a NUMNAME, replace with actual value
            if simpler[3] == 10:
                threeline[2] = word_to_num[threeline[2]]
        else:
            threeline[2] = disthreeline[2]

    # Use the now-numeric value in threeline[2] for indents
    # Only actual control flow: WHILE, ITERATE, COMPARE, ELSECOMPARE, WHILETRUE, IFTRUE, ELSEIF
    # NOT BASICCONDITION, CHANGECOMPARE (these aren't control flow, their arg2 isn't a line number)
    control_flow = {"WHILE", "ITERATE", "COMPARE", "ELSECOMPARE", "WHILETRUE", "IFTRUE", "ELSEIF"}
    if threeline[0] in control_flow:
        listofindents[counter] = 1
        listofindents[int(threeline[2])] = -1
    elif threeline[0] == "DEFINE":
        listfunctions.append(threeline[0])
        listofindents[int(counter)] = 1
    counter += 1
    if len(threeline[1]) == 0 or len(threeline[2]) == 0:
        problem = True
    return threeline