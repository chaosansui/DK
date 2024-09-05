from poliastro.examples import molniya, iss
from poliastro.czml.extract_czml import CZMLExtractor
from czml3.core import Document
from satellitetest.test import create
from satellitetest.specific_time import specific_time



# 定义开始和结束的时间
start_epoch = specific_time
end_epoch = specific_time + 90
sample_points = 10  # 样本点的数量，越多越精确

blue_orb, blueacc_orb, red_orb, delta_v = create()
# 创建 CZML 提取器
extractor = CZMLExtractor(start_epoch, end_epoch, sample_points)

# 添加轨道数据
extractor.add_orbit(blue_orb, label_text="侦察卫星")
extractor.add_orbit(blueacc_orb, label_text="敌方干扰")
extractor.add_orbit(red_orb,label_text="我方干扰")

# 创建 CZML 文档
doc = Document(values=[])

# 将提取的数据包添加到文档中
for packet in extractor.packets:
    doc.packets.append(packet)

# 尝试保存为 CZML 文件
try:
    # 将数据写入到 CZML 文件中
    with open('output.czml', 'w') as f:
        f.write(str(doc))  # 将 CZML 文档转换为字符串并写入文件
    print("CZML data has been written to output.czml")
except Exception as e:
    print(f"Error: {e}")
