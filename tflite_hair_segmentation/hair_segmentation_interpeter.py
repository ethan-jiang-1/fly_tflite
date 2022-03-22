import os
import cv2 as cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from datetime import datetime 

HSI_IMG_SIZE = 512

class HairSegmentationInterpreter(object):
    def __init__(self, debug=False):
        model_pathname = self._get_model_pathname()
        self.ipt = tflite.Interpreter(model_path=model_pathname)
        self.ipt.allocate_tensors()
        self.debug = debug

    def _get_model_pathname(self):
        this_dir = os.path.dirname(__file__)
        paths = [this_dir, "hair_segmentation.tflite"]
        full_path = os.sep.join(paths)
        if not os.path.isfile(full_path):
            raise ValueError("{} not exist".format(full_path))
        return full_path

    def validate(self):
        ipt = self.ipt
        if self.debug: 
            self._debug_ipt(ipt)

        return True

    def _debug_ipt(self, ipt):
        print("== Input details ==")
        inputs = ipt.get_input_details()
        for idx in range(len(inputs)):
            input = inputs[idx]
            print("DUMP INPUT", idx, input)
            print("index:  ", input['index'])
            print("name:   ", input['name'])
            print("shape:  ", input['shape'])
            print("type:   ", input['dtype'])
            print("shape_s:", input["shape_signature"])

        print("\n== Output details ==")
        outputs = ipt.get_output_details()
        for idx in range(len(outputs)):
            output = outputs[idx]
            print("DUMP OUTPUT", idx, output)
            print("index:  ", output['index'])
            print("name:   ", output['name'])
            print("shape:  ", output['shape'])
            print("type:   ", output['dtype'])
            print("shape_s:", output["shape_signature"])

    def _debug_img(self, img):
        cv2.imshow('Example - Show image in window',img)
        
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image

    def _debug_imgs(self, img, mask_black, mask_white):
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        fig.add_subplot(1, 3, 2)
        plt.imshow(mask_black)
        fig.add_subplot(1, 3, 3)
        plt.imshow(mask_white)
        plt.show()

    def process_img512(self, img_cv512):
        #if self.debug:
        #    self._debug_img(img_cv512)

        if img_cv512.shape[0] != HSI_IMG_SIZE and img_cv512.shape[1] != HSI_IMG_SIZE and img_cv512.shape[2] != 3:
            raise ValueError("expecting (512, 512, 3) instead of {}".format(img_cv512))

        d0 = datetime.now()
        img_cv512 = img_cv512 / 256

        img_tfl512 = np.zeros((1, HSI_IMG_SIZE, HSI_IMG_SIZE, 4), dtype=np.float32)    
        img_tfl512[0, :, :, 0:3] = img_cv512

        inputs = self.ipt.get_input_details()
        idx_input = inputs[0]['index']
        self.ipt.set_tensor(idx_input, img_tfl512)   

        self.ipt.invoke()

        outputs = self.ipt.get_output_details()
        idx_output = outputs[0]['index']
        result_masks = self.ipt.get_tensor(idx_output)

        mask_black = np.zeros((HSI_IMG_SIZE, HSI_IMG_SIZE, 1), dtype=np.int32)
        mask_white = np.zeros((HSI_IMG_SIZE, HSI_IMG_SIZE, 1), dtype=np.float32)

        mask_black = result_masks[0, :, :, 0]
        mask_white = result_masks[0, :, :, 1]

        d1 = datetime.now()
        dt = d1 - d0
        print("inference time: {:.3f}sec".format(dt.total_seconds()))
        if self.debug:
            self._debug_imgs(img_cv512, mask_black, mask_white)

        return mask_black, mask_white


def _get_parent_dir():
    import os 
    dir_this = os.path.dirname(__file__)
    dir_parent = os.path.dirname(dir_this)
    return dir_parent


def _add_parent_in_sys_path():
    import sys
    dir_parent = _get_parent_dir()
    if dir_parent not in sys.path:
        sys.path.append(dir_parent)

def do_validate():
    hsi = HairSegmentationInterpreter(debug=True)
    hsi.validate()


def do_process_reseved_img():
    #img_pathname = os.sep.join([_get_parent_dir(), "_reserved_imgs", "c1.png"])
    img_pathname = os.sep.join([_get_parent_dir(), "_reserved_imgs", "c2.jpg"])
    #img_pathname = os.sep.join([_get_parent_dir(), "_reserved_imgs", "c3.png"])

    if not os.path.isfile(img_pathname):
        raise ValueError("{} not exist".format(img_pathname))
    img = cv2.imread(img_pathname, cv2.IMREAD_ANYCOLOR)
    print(img.shape)
    img_resized = cv2.resize(img, (HSI_IMG_SIZE, HSI_IMG_SIZE))
    print(img_resized.shape)
    
    hsi = HairSegmentationInterpreter(debug=True)
    hsi.validate()
    hsi.process_img512(img_resized)


if __name__ == '__main__':
    _add_parent_in_sys_path()
    do_process_reseved_img()
