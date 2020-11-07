'''
    parse pascal VOC xml

    @author neucrack
    @license MIT © 2020 neucrack
'''




import re

def decode_pascal_voc_xml(xml):
    '''
        @reuturn bool, info
                    res = {
                            "filename": ,
                            "path": ,
                            "width": ,
                            "height": ,
                            "depth": ,
                            "bboxes": [(xmin, ymin, xmax, ymax, label, difficult)]
                        }
    '''
    try:
        rule = "<filename>(.*)</filename>.*<path>(.*)</path>.*<size>.*<width>(.*)</width>.*<height>(.*)</height>.*<depth>(.*)</depth>.*</size>"
        match = re.findall(rule, xml, re.MULTILINE|re.DOTALL)
        if len(match) < 1:
            return False, "decode error"
        res = {
            "filename": match[0][0],
            "path": match[0][1],
            "width": int(match[0][2]),
            "height": int(match[0][3]),
            "depth": int(match[0][4]),
            "bboxes": []
        }
        rule = "<object>.*?<name>(.*?)</name>.*?<difficult>(.*?)</difficult>.*?<bndbox>.*?<xmin>(.*?)</xmin>.*?<ymin>(.*?)</ymin>.*?<xmax>(.*?)</xmax>.*?<ymax>(.*?)</ymax>.*?</bndbox>.*?</object>"
        match = re.findall(rule, xml, re.MULTILINE|re.DOTALL)
        if len(match) < 1:
            return False, "no object in this iamge"
        for bbox in match:
            bbox = [int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), bbox[0], int(bbox[1])]
            res["bboxes"].append(bbox)
    except Exception:
        return False, "decode error"
    return True, res

