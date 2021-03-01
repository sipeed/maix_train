'''
    parse pascal VOC xml

    @author neucrack
    @license MIT © 2020 neucrack
'''




import re


def decode_pascal_voc_xml(xml_path, ordered = True):
    '''
        @ordered parse labelimg ordered xml by RE, or will use xml parser
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
    if ordered:
        with open(xml_path) as f:
            xml = f.read()
        try:
            rule = "<filename>(.*)</filename>.*<path>(.*)</path>.*<size>.*<width>(.*)</width>.*<height>(.*)</height>.*<depth>(.*)</depth>.*</size>"
            match = re.findall(rule, xml, re.MULTILINE|re.DOTALL)
            if len(match) < 1:
                return False, "decode error"
            res = {
                "filename": match[0][0].replace("\\", "/"),
                "path": match[0][1].replace("\\", "/"),
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
        except Exception as e:
            return False, "decode error： {}".format(e)
        return True, res
    try:
        from xml.etree.ElementTree import parse
        tree = parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text
        path = root.find("path")
        width = -1
        height = -1
        depth = -1
        for elem in tree.iter():
            if "width" in elem.tag:
                width = int(elem.text)
            elif "height" in elem.tag:
                height = int(elem.text)
            elif "depth" in elem.tag:
                depth = int(elem.text)
        obj_tags = root.findall("object")
        res = {
            "filename": filename,
            "path": filename if path is None else path.text,
            "width": width,
            "height": height,
            "depth": depth,
            "bboxes": []
        }
        res["filename"] = res["filename"].replace("\\", "/"),
        res["path"] = res["path"].replace("\\", "/"),
        for t in obj_tags:
            name = t.find("name").text
            box_tag = t.find("bndbox")
            difficult = int(t.find("difficult").text)
            x1 = int(float(box_tag.find("xmin").text))
            y1 = int(float(box_tag.find("ymin").text))
            x2 = int(float(box_tag.find("xmax").text))
            y2 = int(float(box_tag.find("ymax").text))
            bbox = [x1, y1, x2, y2, name, difficult ]
            res["bboxes"].append(bbox)
        return True, res
    except Exception as e:
        return False, "decode error： {}".format(e)
