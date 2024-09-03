# Project Design

- From this point onwards, progress needs to be made in all 3 branches together (OCR, language, app).
- However, OCR and language need to remain independent to be easily debuggable.
- Currently, the language is ready for deployment in the app. With further progress, more work will be required mainly on the compiler and error-correction, with perhaps some tweaking and bug fixes for the runtime.
- OCR can be split into 3 parts (preprocessing, detection, recognition). Surprisingly, recognition is the easiest part and can be easily tackled by training a TrOCR model for the task with very good accuracy. Preprocessing involves a lot of technical work but shouldn't prove to be an obstacle. Preprocessing requires close integration in some parts with the GUI for testing.
- Detection is the hardest to implement as a good solution currently doesn't appear to exist.
# Missions:
1. merge 3 parts to one directory on my computer with 3 smaller directories inside(git ignore on data))
2. generate training data from captachs and more datasets and finish data set
3. create efficent training pipeline that can be easily stoppedd and resumed(maybe regenerate captchas instead of reusing)

## OCR

### Parts:
1. Preprocessing algorithms for all conditions (color known/unknown, page tilt, shadows, etc.)
2. Text detection algorithm specialized for this type of OCR
3. Text recognition based on TrOCR (works well already)

### Strategy:
- Build as large as possible dataset for text recognition
- Try to find text detection algorithm and train it on custom dataset (hard)
- Try to create algorithm (DL) to determine which preprocessing steps to take (very hard)

## GUI

### Parts:
1. Photo editing with adjustable filters and ability to see detection output on image
2. Send to detection that shows detection output with ability to change lines, words, and rotation
3. Send to recognition that shows original output and then applies error-correction with ability to edit output
4. Ability to execute and edit code

### Strategy:
- Find existing open-source GUI to work with
- Add multicore support: one core for GUI, one core runs the algorithms + GPU for OCR
- Build to be compatible with non-GUI tool

## Language

### Parts:
1. Error-correction using custom Levenshtein? distance
2. Compiler Tzefa to Python
3. Tzefa runtime
4. In the future, ability to debug and define prewritten tests
5. Could be cool to add compatibility with Python code

### Strategy:
- Fix bugs with runtime/compiler/error correction
- Update language based on OCR capabilities (more or less than 3 words in line, small letters)

## Missions

- [ ] Implement interfaces with OCR and Tzefa
- [ ] Implement functions useful for the GUI
- [ ] Complete first full run
- [ ] Add multiprocessing to improve UX
- [ ] Rewrite algorithms for cv2 and also faster performance
- [ ] Reconsider which algorithms to keep and which to delete

## Notes

- Strategy: Main UI window opens processes to run compute-heavy tasks
- Tactic: Send data to shared input array and UI acts upon that data
- Tactic: Manage operations in order
