# Press the green button in the gutter to run the script.
import ErrorCorrection
import topy
import cProfile
import pstats
from pstats import SortKey
def main():
    textlist = [
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