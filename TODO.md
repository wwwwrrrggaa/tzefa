## Project design:

-from this point onwards progress needs to be made in all 3 branches together(ocr,language,app).
-however ocr and language need to remain independent in order to be easily debuggable
-currently the language is ready for deployment in app, with further progress more work will be required mainly on the
compiler and error-correction with perhaps some tweaking and bux fixes for the runtime
-ocr can be split to 3 parts(preprocessing,detection,recognition).somehow recognition is the easiest part and can be
easily tackled with training a trocr model for the task with very good accuracy, preprocessing is a lot of technical
work but shouldn't prove to be an obstacle.
preprocessing requires close integration in some parts with the gui for testing
detection is the hardest to implement a good solution as one currently doesn't appear to exist.

## ocr:

    parts:
        1 preprocessing algorithms for all conditions(color known/unknown page tilt shadows and so on)
        2 text detection algorithm speaclized for this type of ocr
        3 text recognition based on trocr(works well already)
    strategy:
        build as large as possible dataset for text recognition
        try to find text detection algorithm and train it on custom dataset(hard)
        try to create algorithm(dl) to determine which preprocessing steps to take(very hard)

## gui:

    parts:
        1 photoediting with adjustable filters and ability to see detecion output on image
        2 send to detection that shows detection output with ability to change lines words and rotation
        3 send to recognition that shows original output and then applies error-correction with ability to edit output
        4 ability to execute and edit code
    strategy:
        find existing opensource gui to work with
        add multicore support one core gui one core runs the algorithms+gpu for ocr
        build to be compatible with non gui tool

## language:

    parts:
        1 error-correction using custom levnstien? distance
        2 compiler tzefa to python
        3 tzefa runtime
        4 in the future ability to debug and define prewritten tests
        5 could be cool to add compatibility with python code
    strategy:
        fix bugs with runtime/compiler/errorcorrection
        update language based on ocr capabilities (more or less than 3 words in line, small letters


## Missions

- [ ] implement interfaces with ocr and tzefa
- [ ] implement functions useful for the gui
- [ ] complete first full run
- [ ] add multiprocessing to improve ux
- [ ] rewrite algorithms for cv2 and also faster perforamnce
- [ ] reconsider which algorithms to keep and which to delete

## Notes

- strategy: main ui windows opens processes to run compute heavy tasks
- tactic:  send data to shared input array and ui acts upon that data
- tactic:  manage operations in order