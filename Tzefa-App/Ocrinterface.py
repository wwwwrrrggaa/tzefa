from mmocr.apis import MMOCRInferencer

ocr = MMOCRInferencer(det='DBNetpp')
ocr(r"E:\Storage\tests\PXL_20240525_142704542.MP.jpg", show=True, print_result=True)
