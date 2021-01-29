# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

from importlib import import_module
import json

CLASS_KEY = 'CLASS'
MODULE_KEY = 'MODULE'

def save_object_param(obj, params, file_path):
    params[CLASS_KEY] = obj.__class__.__name__
    params[MODULE_KEY] = obj.__class__.__module__
    with open(file_path, 'w') as jfile:
        json.dump(params, jfile)

def load_object_param(file_name):
    with open(file_name, 'r') as jfile:
        params = json.load(jfile)
        if CLASS_KEY not in params: return None
        cls_name = params[CLASS_KEY]
        module = params[MODULE_KEY]

    cls = getattr(import_module(module), cls_name)
    params.pop(CLASS_KEY)
    params.pop(MODULE_KEY)
    obj = cls(**params)
    return obj, cls_name, module, params

def load_object(cls_name, module, params):
    cls = getattr(import_module(module), cls_name)
    return cls(**params)
