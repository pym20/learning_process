import os
import random
from lxml import etree
import json
# 从xml里面提取类别信息
# xml解析函数
def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def main():
    # voc_root = "D:\yanjiusheng\数据集\HelmetDetection"
    random.seed(0)  # 设置随机种子，保证随机结果可复现
    # 文件根目录
    files_path = "D:\yanjiusheng\数据集\HelmetDetection\Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5  #  验证集比例
    # listdir：返回一个包含目录中文件名称的列表。
    # S.split(sep=None, maxsplit=-1)：返回S中的单词列表，使用sep作为分隔符的字符串。
    # 排序
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    # k为采样个数，val_index得到的是索引值，下面通过索引值来判断是不是value
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)

        else:
            train_files.append(file_name)
# 从Annotations构建类别json
    labels = dict()
    i=1
    Imagelist = os.listdir(files_path)
    for index in range(0, len(Imagelist)):
        xml_path = os.path.join(files_path, Imagelist[index])
        with open(xml_path) as fid:
            xml_str = fid.read()
        # fromstring：从字符串中解析XML文档或片段，返回根节点(或解析器目标返回的结果)。
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]  # 解析xml文件，以字典方式存储

        for obj in data["object"]:
            if obj["name"] not in labels.keys():
                labels[obj["name"]] = i
                i += 1


    try:
        # 相对路径
        # train_f = open("train.txt", "x")
        # eval_f = open("val.txt", "x")
        # x，代表创建一个新文件并打开它进行写入,绝对路径
        train_f = open(r"D:\yanjiusheng\数据集\HelmetDetection\ImageSets\Main\train.txt", "x")
        eval_f = open(r"D:\yanjiusheng\数据集\HelmetDetection\ImageSets\Main\val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        json_dict = json.dumps(labels)
        file = open("pascal_voc_classes.json","w")
        file.write(json_dict)
        file.close()

    except FileExistsError as e:
        print(e)
        exit(1)



if __name__ == '__main__':
    main()
