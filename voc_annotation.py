import xml.etree.ElementTree as ET
from os import getcwd
wd = getcwd()
labels = ['cat','dog']

def convert_annotation(image_id,list_file):
    dir = open('dataset/annotations/%s.xml'%(image_id))
    t = ET.parse(dir)
    root = t.getroot()
    if root.find('object') == None:
        return
    list_file.write('%s/dataset/images/%s.png'%(wd, image_id))
    for x in root.iter('object'):
        name = x.find('name').text
        dif = x.find('difficult').text
        if name not in labels or int(dif) == 1:
            continue
        xmlbox = x.find('bndbox')
        cls = labels.index(name)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(' '+','.join([str(a) for a in b]) + ','+str(cls))
    list_file.write('\n')

list_file = open('label.txt','w')
for i in range(3686):
    image_id = 'Cats_Test'+str(i)
    convert_annotation(image_id,list_file)
list_file.close()
