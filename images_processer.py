import os
import sys
import copy
import cv2

class InputError(Exception):

    def __init__(self, message):
        self.message = message

class OpenError(Exception):

    def __init__(self, message):
        self.message = message

class BaseClass:

    def read():
        raise NotImplementedError

    def type():
        raise NotImplementedError


class ImageReader(BaseClass):

    def __init__(self, input, loop):
        self.loop = loop
        if not os.path.isfile(input):
            raise InputError(f"Can't find the image by {input}")
        self.image = cv2.imread(input, cv2.IMREAD_COLOR)
        self.can_read = True
        if self.image is None:
            raise OpenError(f"Can't open the image from {input}")

    def read(self):
        if self.loop:
            return copy.deepcopy(self.image)
        if self.can_read:
            self.can_read = False
            return copy.deepcopy(self.image)
        return None

    def type(self):
        return 'IMAGE'


class DirReader(BaseClass):

    def __init__(self, input, loop):
        self.loop = loop
        if not os.path.isdir(input):
            raise InputError(f"Can't find the dir by {input}")
        self.names = os.listdir(input)
        if not self.names:
            raise OpenError(f'The dir {input} is empty')
        self.file_id = 0

    def read(self):
        while self.file_id < len(self.names):
            filename = os.path.join(self.dir, self.names[self.file_id])
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.file_id += 1
            if image is not None:
                return image
        
        if self.loop:
            self.file_id = 0
            while self.file_id < len(self.names):
                filename = os.path.join(self.dir, self.names[self.file_id])
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                self.file_id += 1
                if image is not None:
                    return image
        return None

    def type(self):
        return 'DIRECTORY'


def open_images(input, loop):
    errors = {InputError: [], OpenError: []}
    for reader in (ImageReader, DirReader):
        try:
            return reader(input, loop)
        except (InputError, OpenError) as exc:
            errors[type(exc)].append(exc.message)

    if not errors[OpenError]:
        print(*errors[InputError], file=sys.stderr, sep='\n')
    else:
        print(*errors[OpenError], file=sys.stderr, sep='\n')
    sys.exit(1)
