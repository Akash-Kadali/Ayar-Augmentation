import os

sample_names = [
    'capsule',
    'bottle',
    'carpet',
    'leather',
    'pill',
    'transistor',
    'tile',
    'cable',
    'zipper',
    'toothbrush',
    'metal_nut',
    'hazelnut',
    'screw',
    'grid',
    'wood'
]
import csv

with open("IS.csv", "w") as csvfile:
    pass

for sample_name in sample_names:
    dir_name = 'generate_data_dir/%s/' % sample_name
    dis = 0
    cnt = 0
    for anomaly_name in os.listdir(dir_name):
        print(sample_name,anomaly_name)
        os_str = 'fidelity --gpu 0 --isc --input1 %s/%s/image' % (dir_name, anomaly_name)
        f = os.popen(os_str, 'r')
        res = f.readlines()[0]  # res接受返回结果
        f.close()
        print(res)
        data=res[res.index(':')+1:-1]
        print(data)
        dis += float(data)
        cnt += 1
    with open("IS.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sample_name, str(float(dis / cnt))])
