# Press the green button in the gutter to run the script.
import ErrorCorrection
import topy
import cProfile
import pstats
from pstats import SortKey
def main():
    textlist = [
    "MAKEINTEGER INTEGERONE 32",
    "MAKEINTEGER INTEGERTWO 4",
    "NEWLIST GCDLIST TWO",
    "SETINDEX GCDLIST ZERO ",
    "WRITEINTEGER GCDLIST INTEGERONE",
    "SETINDEX GCDLIST ONE ",
    "WRITEINTEGER GCDLIST INTEGERWO",
    "BASICCONDITION GCDCOMPARE BIGGER",
    "LEFESIDE GCDCOMPARE INTEGERTWO",
    "RIGHTSIDE GCDCOMPARE ZERO",


    "LISTFUNCTION GCD GCBLIST",
    "SETINDEX GCDLIST ZERO ",
    "GETINTEGER GCDLIST INTEGERONE",
    "SETINDEX GCDLIST ONE ",
    "GETINTEGER GCDLIST INTEGERTWO",
    "MODULU INTEGERONE INTEGERTWO",
    "ASSIGNINT INTGERONE INTEGERTWO",
    "ASSIGNINT INTGERTWO TEMPORARY",
    "COMPARE GCDCOMPARE 24",
    "SETINDEX GCDLIST ZERO ",
    "WRITEINTEGER GCDLIST INTEGERONE",
    "SETINDEX GCDLIST ONE ",
    "WRITEINTEGER GCDLIST INTEGERWO",
    "GCD GCDLIST GCDLIST",
    "RETURN GCDLIST BREAK",

    "GCD GCDLIST GCDLIST",

    ]
    listofindentchanges = [0] * (len(textlist) + 1)

    compiledlinelist=[ErrorCorrection.toline
                      (textlist[lineindex],ErrorCorrection.handelfirstword(textlist[lineindex].split(" ")[0])[1],listofindentchanges)
                      for lineindex in range(len(textlist))]
    listfunctions,listezfunctions=ErrorCorrection.giveinstructions()
    topy.getinstructions(listfunctions,listezfunctions)
    topy.makepyfile(compiledlinelist)
    import test
def compile_and_run(textlist):
    listofindentchanges = [0] * (len(textlist) + 1)

    compiledlinelist = [ErrorCorrection.toline
                        (textlist[lineindex], ErrorCorrection.handelfirstword(textlist[lineindex].split(" ")[0])[1],
                         listofindentchanges)
                        for lineindex in range(len(textlist))]
    listfunctions, listezfunctions = ErrorCorrection.giveinstructions()
    topy.getinstructions(listfunctions, listezfunctions)
    topy.makepyfile(compiledlinelist)
    import test
if __name__ == '__main__':
    cProfile.run('main()', 'output.prof')

    # Print sorted stats
    with open("output_stats.txt", 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats(SortKey.TIME).print_stats()