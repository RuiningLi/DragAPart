import os
import os
from os.path import join as pjoin
import numpy as np

HEIGHT = 512
WIDTH = 512

# TODO: Set the path to the dataset
PARTNET_DATASET_PATH = '/path/to/partnet/root/directory'
AKB48_DATASET_PATH = '/path/to/akb48/root/directory'

# TODO: Set the path to save the Drag-a-Move dataset
SAVE_PATH = '/path/to/save/drag-a-move/dataset'

PARTNET_ID_PATH = pjoin(os.path.dirname(__file__), "..", "meta", "partnet_all_id_list.txt")
AKB48_ID_PATH = pjoin(os.path.dirname(__file__), "..", "meta", "akb48_all_id_list.txt")

TARGET_GAPARTS = [
    'line_fixed_handle', 'round_fixed_handle', 'slider_button', 'hinge_door', 'slider_drawer',
    'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle'
]
TARGET_MOVING_PARTS = [
    'hinge_door', 'slider_drawer', 'hinge_lid', 'hinge_handle', 
]

PARTNET_OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]
AKB48_OBJECT_CATEGORIES = [
    'Box', 'TrashCan', 'Bucket', 'Drawer'
]

PARTNET_CAMERA_POSITION_RANGE = {
    'Box': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'Camera': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }, {
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': -60.0,
        'phi_max': 60.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'CoffeeMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Dishwasher': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'KitchenPot': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Microwave': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Oven': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Phone': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Refrigerator': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Remote': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Safe': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'StorageFurniture': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4.1,
        'distance_max': 5.2
    }],
    'Table': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.8,
        'distance_max': 4.5
    }],
    'Toaster': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'TrashCan': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4,
        'distance_max': 5.5
    }],
    'WashingMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Keyboard': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3,
        'distance_max': 3.5
    }],
    'Laptop': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Door': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Printer': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Suitcase': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Bucket': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Toilet': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }]
}
AKB48_CAMERA_POSITION_RANGE = {
    'Box': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.35,
        'distance_max': 0.55
    }],
    'TrashCan': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.35,
        'distance_max': 0.55
    }],
    'Bucket': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.45,
        'distance_max': 0.6
    }],
    'Drawer': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.55,
        'distance_max': 0.8
    }]
}

BACKGROUND_RGB = np.array([255, 255, 255], dtype=np.uint8)
