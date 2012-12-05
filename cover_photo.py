#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import sys
import logging
import tempfile
import urllib2
import os
import time
import random
import json
import traceback
import gzip
import cv

from PIL import Image,ImageStat


class CoverPhoto():
    def __init__(self, opencv_cascade_dir):
        #cascade_xml = 'haarcascade_frontalface_default.xml'
        cascade_xml = 'haarcascade_frontalface_alt.xml'
        self.cascade = cv.Load(os.path.join(opencv_cascade_dir, cascade_xml))

    def compute_is_bright(self, fh):
        fh.seek(0)
        img = Image.open(fh)
        gsimg = img.convert(mode='L')
        im_stat = ImageStat.Stat(gsimg) 
        median = im_stat.median[0]

        cutoff = 60
        if median < cutoff:
            logging.info('Image has brightness level %d (dark)' % (median))
            return False
        else:
            logging.info('Image has brightness level %d (bright)' % (median))
            return True

    def compute_num_faces_from_url(self, url):
        num_faces = None
        is_bright = None
        num_retries = 0
        with tempfile.NamedTemporaryFile(dir='/tmp/') as fh:
            while num_faces is None and num_retries < 3:
                try:
                    logging.info(url)
                    urlfile = urllib2.urlopen(url, timeout=10)
                    fh.write(urlfile.read())
                    fh.flush()

                    is_bright = self.compute_is_bright(fh)
                    image = cv.LoadImageM(fh.name, cv.CV_LOAD_IMAGE_GRAYSCALE)
                    num_faces = self.compute_num_faces(image)
                    logging.info('Number of faces: %d' % num_faces)
                except Exception as e:
                    logging.error('Exception caught: %s' % str(e))
                    traceback.print_exc()
                    time.sleep(2 ** (num_retries+1))
                num_retries += 1
                
        if num_faces:
            logging.info('Face found in %s' % url)

        if num_faces is None or is_bright is None: 
            return None

        return {
                'num_faces' : num_faces,
                'is_bright' : is_bright,
                }

    def compute_num_faces(self, image):
        min_len = min(image.rows, image.cols) // 10

        #For more details, see:
        #http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html

        faces = cv.HaarDetectObjects(
                image, 
                self.cascade, 
                storage=cv.CreateMemStorage(0), 
                min_size=(min_len, min_len),

                #FAST params
                #scale_factor=1.2, 
                #min_neighbors=2, 
                #flags=cv.CV_HAAR_DO_CANNY_PRUNING, 

                #SLOW params
                scale_factor=1.1, 
                min_neighbors=3, 
                flags=0,
                )
        return len(faces)

if __name__ == '__main__':
    epilog= """

Examples:
    """
    argp = ArgumentParser(epilog=epilog)
    args = argp.parse_args()

    print('CMD: ' + ' ' . join(sys.argv), file=sys.stderr)

    


