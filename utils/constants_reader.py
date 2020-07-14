import json

constant_path = './constants/constants.json'


class Constant(object):
    '''
    For reading constants from json file into python object
    '''
    
    @classmethod
    def __init__(cls):
        with open(constant_path) as f:
            json_object =  json.load(f)

        cls.constant_dict =  dict()

        for const in json_object.keys():
            cls.constant_dict[const] = json_object.get(const, None)

    
    @classmethod
    def __getattr__(cls, name):
        return cls.constant_dict.get(name, None)