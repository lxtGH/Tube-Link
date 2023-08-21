import json

json_path = "/data/data1/datasets/mapillary/annotations/panoptic_train.json"

file_content = json.load(open(json_path, "r"))

print(file_content.keys())
# print(file_content['categories'])

cate_list = file_content['categories']

thing_labels = []
stuff_labels = []
for cat in cate_list:
    if cat["isthing"] == 0:
        stuff_labels.append(cat['name'])
    else:
        thing_labels.append(cat['name'])
#
# print("Thing:", len(thing_labels))
# print(thing_labels)
# print("Stuff:", len(stuff_labels))
# print(stuff_labels)


print("cate:")
# print(cate_list)


map_cats = [{'supercategory': 'animal--bird', 'color': [165, 42, 42], 'isthing': 1, 'id': 1, 'name': 'Bird'},
        {'supercategory': 'animal--ground-animal', 'color': [0, 192, 0], 'isthing': 1, 'id': 2, 'name': 'Ground Animal'},
        {'supercategory': 'construction--barrier--curb', 'color': [196, 196, 196], 'isthing': 0, 'id': 3, 'name': 'Curb'},
        {'supercategory': 'construction--barrier--fence', 'color': [190, 153, 153], 'isthing': 0, 'id': 4, 'name': 'Fence'},
        {'supercategory': 'construction--barrier--guard-rail', 'color': [180, 165, 180], 'isthing': 0, 'id': 5, 'name': 'Guard Rail'},
        {'supercategory': 'construction--barrier--other-barrier', 'color': [90, 120, 150], 'isthing': 0, 'id': 6, 'name': 'Barrier'},
        {'supercategory': 'construction--barrier--wall', 'color': [102, 102, 156], 'isthing': 0, 'id': 7, 'name': 'Wall'},
        {'supercategory': 'construction--flat--bike-lane', 'color': [128, 64, 255], 'isthing': 0, 'id': 8, 'name': 'Bike Lane'},
        {'supercategory': 'construction--flat--crosswalk-plain', 'color': [140, 140, 200], 'isthing': 1, 'id': 9, 'name': 'Crosswalk - Plain'},
        {'supercategory': 'construction--flat--curb-cut', 'color': [170, 170, 170], 'isthing': 0, 'id': 10, 'name': 'Curb Cut'},
        {'supercategory': 'construction--flat--parking', 'color': [250, 170, 160], 'isthing': 0, 'id': 11, 'name': 'Parking'},
        {'supercategory': 'construction--flat--pedestrian-area', 'color': [96, 96, 96], 'isthing': 0, 'id': 12, 'name': 'Pedestrian Area'},
        {'supercategory': 'construction--flat--rail-track', 'color': [230, 150, 140], 'isthing': 0, 'id': 13, 'name': 'Rail Track'},
        {'supercategory': 'construction--flat--road', 'color': [128, 64, 128], 'isthing': 0, 'id': 14, 'name': 'Road'},
        {'supercategory': 'construction--flat--service-lane', 'color': [110, 110, 110], 'isthing': 0, 'id': 15, 'name': 'Service Lane'},
        {'supercategory': 'construction--flat--sidewalk', 'color': [244, 35, 232], 'isthing': 0, 'id': 16, 'name': 'Sidewalk'},
        {'supercategory': 'construction--structure--bridge', 'color': [150, 100, 100], 'isthing': 0, 'id': 17, 'name': 'Bridge'},
        {'supercategory': 'construction--structure--building', 'color': [70, 70, 70], 'isthing': 0, 'id': 18, 'name': 'Building'},
        {'supercategory': 'construction--structure--tunnel', 'color': [150, 120, 90], 'isthing': 0, 'id': 19, 'name': 'Tunnel'},
        {'supercategory': 'human--person', 'color': [220, 20, 60], 'isthing': 1, 'id': 20, 'name': 'Person'},
        {'supercategory': 'human--rider--bicyclist', 'color': [255, 0, 0], 'isthing': 1, 'id': 21, 'name': 'Bicyclist'},
        {'supercategory': 'human--rider--motorcyclist', 'color': [255, 0, 100], 'isthing': 1, 'id': 22, 'name': 'Motorcyclist'},
        {'supercategory': 'human--rider--other-rider', 'color': [255, 0, 200], 'isthing': 1, 'id': 23, 'name': 'Other Rider'},
        {'supercategory': 'marking--crosswalk-zebra', 'color': [200, 128, 128], 'isthing': 1, 'id': 24, 'name': 'Lane Marking - Crosswalk'},
        {'supercategory': 'marking--general', 'color': [255, 255, 255], 'isthing': 0, 'id': 25, 'name': 'Lane Marking - General'},
        {'supercategory': 'nature--mountain', 'color': [64, 170, 64], 'isthing': 0, 'id': 26, 'name': 'Mountain'},
        {'supercategory': 'nature--sand', 'color': [230, 160, 50], 'isthing': 0, 'id': 27, 'name': 'Sand'},
        {'supercategory': 'nature--sky', 'color': [70, 130, 180], 'isthing': 0, 'id': 28, 'name': 'Sky'},
        {'supercategory': 'nature--snow', 'color': [190, 255, 255], 'isthing': 0, 'id': 29, 'name': 'Snow'},
        {'supercategory': 'nature--terrain', 'color': [152, 251, 152], 'isthing': 0, 'id': 30, 'name': 'Terrain'},
        {'supercategory': 'nature--vegetation', 'color': [107, 142, 35], 'isthing': 0, 'id': 31, 'name': 'Vegetation'},
        {'supercategory': 'nature--water', 'color': [0, 170, 30], 'isthing': 0, 'id': 32, 'name': 'Water'},
        {'supercategory': 'object--banner', 'color': [255, 255, 128], 'isthing': 1, 'id': 33, 'name': 'Banner'},
        {'supercategory': 'object--bench', 'color': [250, 0, 30], 'isthing': 1, 'id': 34, 'name': 'Bench'},
        {'supercategory': 'object--bike-rack', 'color': [100, 140, 180], 'isthing': 1, 'id': 35, 'name': 'Bike Rack'},
        {'supercategory': 'object--billboard', 'color': [220, 220, 220], 'isthing': 1, 'id': 36, 'name': 'Billboard'},
        {'supercategory': 'object--catch-basin', 'color': [220, 128, 128], 'isthing': 1, 'id': 37, 'name': 'Catch Basin'},
        {'supercategory': 'object--cctv-camera', 'color': [222, 40, 40], 'isthing': 1, 'id': 38, 'name': 'CCTV Camera'},
        {'supercategory': 'object--fire-hydrant', 'color': [100, 170, 30], 'isthing': 1, 'id': 39, 'name': 'Fire Hydrant'},
        {'supercategory': 'object--junction-box', 'color': [40, 40, 40], 'isthing': 1, 'id': 40, 'name': 'Junction Box'},
        {'supercategory': 'object--mailbox', 'color': [33, 33, 33], 'isthing': 1, 'id': 41, 'name': 'Mailbox'},
        {'supercategory': 'object--manhole', 'color': [100, 128, 160], 'isthing': 1, 'id': 42, 'name': 'Manhole'},
        {'supercategory': 'object--phone-booth', 'color': [142, 0, 0], 'isthing': 1, 'id': 43, 'name': 'Phone Booth'},
        {'supercategory': 'object--pothole', 'color': [70, 100, 150], 'isthing': 0, 'id': 44, 'name': 'Pothole'},
        {'supercategory': 'object--street-light', 'color': [210, 170, 100], 'isthing': 1, 'id': 45, 'name': 'Street Light'},
        {'supercategory': 'object--support--pole', 'color': [153, 153, 153], 'isthing': 1, 'id': 46, 'name': 'Pole'},
        {'supercategory': 'object--support--traffic-sign-frame', 'color': [128, 128, 128], 'isthing': 1, 'id': 47, 'name': 'Traffic Sign Frame'},
        {'supercategory': 'object--support--utility-pole', 'color': [0, 0, 80], 'isthing': 1, 'id': 48, 'name': 'Utility Pole'},
        {'supercategory': 'object--traffic-light', 'color': [250, 170, 30], 'isthing': 1, 'id': 49, 'name': 'Traffic Light'},
        {'supercategory': 'object--traffic-sign--back', 'color': [192, 192, 192], 'isthing': 1, 'id': 50, 'name': 'Traffic Sign (Back)'},
        {'supercategory': 'object--traffic-sign--front', 'color': [220, 220, 0], 'isthing': 1, 'id': 51, 'name': 'Traffic Sign (Front)'},
        {'supercategory': 'object--trash-can', 'color': [140, 140, 20], 'isthing': 1, 'id': 52, 'name': 'Trash Can'},
        {'supercategory': 'object--vehicle--bicycle', 'color': [119, 11, 32], 'isthing': 1, 'id': 53, 'name': 'Bicycle'},
        {'supercategory': 'object--vehicle--boat', 'color': [150, 0, 255], 'isthing': 1, 'id': 54, 'name': 'Boat'},
        {'supercategory': 'object--vehicle--bus', 'color': [0, 60, 100], 'isthing': 1, 'id': 55, 'name': 'Bus'},
        {'supercategory': 'object--vehicle--car', 'color': [0, 0, 142], 'isthing': 1, 'id': 56, 'name': 'Car'},
        {'supercategory': 'object--vehicle--caravan', 'color': [0, 0, 90], 'isthing': 1, 'id': 57, 'name': 'Caravan'},
        {'supercategory': 'object--vehicle--motorcycle', 'color': [0, 0, 230], 'isthing': 1, 'id': 58, 'name': 'Motorcycle'},
        {'supercategory': 'object--vehicle--on-rails', 'color': [0, 80, 100], 'isthing': 0, 'id': 59, 'name': 'On Rails'},
        {'supercategory': 'object--vehicle--other-vehicle', 'color': [128, 64, 64], 'isthing': 1, 'id': 60, 'name': 'Other Vehicle'},
        {'supercategory': 'object--vehicle--trailer', 'color': [0, 0, 110], 'isthing': 1, 'id': 61, 'name': 'Trailer'},
        {'supercategory': 'object--vehicle--truck', 'color': [0, 0, 70], 'isthing': 1, 'id': 62, 'name': 'Truck'},
        {'supercategory': 'object--vehicle--wheeled-slow', 'color': [0, 0, 192], 'isthing': 1, 'id': 63, 'name': 'Wheeled Slow'},
        {'supercategory': 'void--car-mount', 'color': [32, 32, 32], 'isthing': 0, 'id': 64, 'name': 'Car Mount'},
        {'supercategory': 'void--ego-vehicle', 'color': [120, 10, 10], 'isthing': 0, 'id': 65, 'name': 'Ego Vehicle'}]


print(file_content['annotations'][0])

print(file_content['annotations'][1])

cat2label = {}
thing_id = 0
stuff_id = 37

for id, cat in enumerate(cate_list):
    if cat["isthing"] == 1: # thing
        cat2label[cat['id']] = thing_id
        thing_id = thing_id + 1
    else:
        cat2label[cat['id']] = stuff_id
        stuff_id = stuff_id + 1

# print(cat2label)
# print(len(cat2label))
# print(set(cat2label.values()))

print("len:", len(file_content['annotations']))

