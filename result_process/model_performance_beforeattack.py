import pandas as pd
import pandas as pd
import json
import pdb
import re

renwu_name=[]
dataset_name=[]
corrutipn_name=[]
corrutipn_level=[]
Model_name = []
value_lengths = []
blip_zhibiao=[]
InstructBLIP_zhibiao=[]
LLaMA_Adapter_v2_zhibiao=[]
LLaVA_zhibiao=[]
MiniGPT_zhibiao=[]
mPLUG_zhibiao=[]
Otter_zhibiao=[]
PandaGPT_zhibiao=[]
VPGTrans_zhibiao=[]
OFV2_zhibiao=[]
xcomposer_zhibiao=[]
LLaVA15_zhibiao=[]
moe_zhibiao=[]
share_zhibiao=[]

nengli={"vis_p":['CLS','MCI'],
"vis_k":['CAP','KIE','OCR'],
"vis_r":['VQA','VQACHOICE'],
"vis_c": ['imagenetvc'],
"OH":['OBJECT'],}


for neng in nengli:
    renwu=nengli[neng]

    for ren in renwu:
        with open('BLIP2_ff/{}/result.json'.format(ren), 'r') as file:
            data_BLIP2 = json.load(file)
        with open('InstructBLIP_ff/{}/result.json'.format(ren), 'r') as file:
            data_InstructBLIP = json.load(file)
        with open('LLaMA-Adapter-v2_ff/{}/result.json'.format(ren), 'r') as file:
            data_LLaMA_Adapter_v2 = json.load(file)

        with open('LLaVA_ff/{}/result.json'.format(ren), 'r') as file:
            data_LLaVA = json.load(file)
        with open('MiniGPT-4_ff/{}/result.json'.format(ren), 'r') as file:
            data_MiniGPT = json.load(file)
        with open('mPLUG-Owl_ff/{}/result.json'.format(ren), 'r') as file:
            data_mPLUG = json.load(file)

        with open('Otter_ff/{}/result.json'.format(ren), 'r') as file:
            data_Otter = json.load(file)
        with open('PandaGPT_ff/{}/result.json'.format(ren), 'r') as file:
            data_PandaGPT = json.load(file)
        with open('VPGTrans_ff/{}/result.json'.format(ren), 'r') as file:
            data_VPGTrans = json.load(file)

        with open('OFv2_ff/{}/result.json'.format(ren), 'r') as file:
            data_OFv2 = json.load(file)
        with open('internlm-xcomposer_ff/{}/result.json'.format(ren), 'r') as file:
            data_xcomposer = json.load(file)
        with open('LLaVA15_ff/{}/result.json'.format(ren), 'r') as file:
            data_LLaVA15 = json.load(file)      
        with open('moellava_ff/{}/result.json'.format(ren), 'r') as file:
            data_moe = json.load(file)
        with open('sharegpt4v_ff/{}/result.json'.format(ren), 'r') as file:
            data_share= json.load(file)
        
            
        for index, (key, value) in enumerate(data_BLIP2.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':         
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value 
            if level==5:   
                renwu_name.append(ren)
                dataset_name.append(key.split("severity_")[0][:-1])
                corrutipn_name.append(key.split("severity_")[1][:-2])
                corrutipn_level.append("0")
            if level==5:               
                blip_zhibiao.append(init)
        for index, (key, value) in enumerate(data_InstructBLIP.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value               
            if level==5:
                InstructBLIP_zhibiao.append(init)
                
        for index, (key, value) in enumerate(data_LLaMA_Adapter_v2.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc'] 
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value 
            if level==5:
                LLaMA_Adapter_v2_zhibiao.append(init)
        for index, (key, value) in enumerate(data_LLaVA.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value                
            if level==5:
                LLaVA_zhibiao.append(init)
        for index, (key, value) in enumerate(data_MiniGPT.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value
            if level==5:
                MiniGPT_zhibiao.append(init)
        for index, (key, value) in enumerate(data_mPLUG.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']               
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value               
            else:
                if level==0:
                    init = value               
            if level==5:
                mPLUG_zhibiao.append(init)
        for index, (key, value) in enumerate(data_Otter.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']             
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value                
            else:
                if level==0:
                    init = value             
            if level==5:
                Otter_zhibiao.append(init)
        for index, (key, value) in enumerate(data_PandaGPT.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']               
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value                
            else:
                if level==0:
                    init = value                
            if level==5:
                PandaGPT_zhibiao.append(init)
        for index, (key, value) in enumerate(data_VPGTrans.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']                
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value               
            if level==5:
                VPGTrans_zhibiao.append(init)
        for index, (key, value) in enumerate(data_OFv2.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value
               
            if level==5:
                OFV2_zhibiao.append(init)
        for index, (key, value) in enumerate(data_xcomposer.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value               
            else:
                if level==0:
                    init = value                
            if level==5:
                xcomposer_zhibiao.append(init)
        for index, (key, value) in enumerate(data_LLaVA15.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']               
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value                
            else:
                if level==0:
                    init = value
            if level==5:
                LLaVA15_zhibiao.append(init)
        

        for index, (key, value) in enumerate(data_share.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']                
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value
            if level==5:
                share_zhibiao.append(init)

        for index, (key, value) in enumerate(data_moe.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']                
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
            else:
                if level==0:
                    init = value
            if level==5:
                moe_zhibiao.append(init)

# 创建DataFrame来存储分析结果
analysis_df = pd.DataFrame({
    'renwu_name': renwu_name,
    'datasets_name': dataset_name,
    'corrutipn_name': corrutipn_name,
    'corrutipn_level': corrutipn_level,
    'BLIP2': blip_zhibiao,
    'InstructBLIP':InstructBLIP_zhibiao,
    'LLaMA-Adapter-v2_f':LLaMA_Adapter_v2_zhibiao,
    'LLaVA': LLaVA_zhibiao,
    'MiniGPT-4': MiniGPT_zhibiao,
    'mPLUG-Owl': mPLUG_zhibiao,
    'Otter': Otter_zhibiao,
    'PandaGPT': PandaGPT_zhibiao,
    'VPGTrans': VPGTrans_zhibiao,
    'OFV2': OFV2_zhibiao,
    'internlm-xcomposer':xcomposer_zhibiao,
    'LLaVA15':LLaVA15_zhibiao,
    'ShareGPT4V':share_zhibiao,
    'Moe-LLaVA':moe_zhibiao,

})
# 将DataFrame保存为Excel文件
analysis_df.to_excel('corruption_result_ys_240305.xlsx', index=False)