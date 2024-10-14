import pandas as pd
import pandas as pd
import json
import pdb
import re
# import string
#BLIP2
# 从JSON文件加载数据

renwu_name=[]
dataset_name=[]
attack_name=[]

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
sharegpt4v_zhibiao=[]
moellava_zhibiao=[]

nengli={"vis_p":{'classification':['CIFAR10','CIFAR100','Flowers102','ImageNet','OxfordIIITPet']},
"vis_k":{'caption':['Flickr','MSCOCO_caption_karpathy','NoCaps','WHOOPSCaption'],'OCR':['COCO-Text','CTW','CUTE80','HOST','IC13','IC15']},
"vis_r":{'vqa':['AOKVQAClose','IconQA','ScienceQAIMG','WHOOPSWeird']},
"OH":{'object':['MSCOCO_pope_adversarial','MSCOCO_pope_popular','MSCOCO_pope_random']},}

attack=['textbugger', 'deepwordbug','pruthi','bertattack', 'textfooler','pwws','checklist', 'stresstest','input-reduction','semantic']


for neng in nengli:
    renwu=nengli[neng]
    for ren in renwu:
        for sett in nengli[neng][ren]:
            for att in attack:
                if att=='semantic':
                    with open('BLIP2_d/{}/{}/BLIP2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_BLIP2 = file.readlines()
                    with open('InstructBLIP_d/{}/{}/InstructBLIP_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_InstructBLIP = file.readlines()
                    with open('LLaMA-Adapter-v2_d/{}/{}/LLaMA-Adapter-v2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaMA_Adapter_v2 = file.readlines()

                    with open('LLaVA_d/{}/{}/LLaVA_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaVA = file.readlines()
                    with open('MiniGPT-4_d/{}/{}/MiniGPT-4_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_MiniGPT = file.readlines()
                    with open('mPLUG-Owl_d/{}/{}/mPLUG-Owl_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_mPLUG = file.readlines()

                    with open('Otter_d/{}/{}/Otter_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_Otter = file.readlines()
                    with open('PandaGPT_d/{}/{}/PandaGPT_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_PandaGPT = file.readlines()
                    with open('VPGTrans_d/{}/{}/VPGTrans_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_VPGTrans = file.readlines()

                    with open('OFv2_d/{}/{}/OFv2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_OFv2 = file.readlines()
                    with open('internlm-xcomposer_d/{}/{}/internlm-xcomposer_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_xcomposer = file.readlines()
                    with open('LLaVA15_d/{}/{}/LLaVA15_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaVA15 = file.readlines()                    
                    with open('sharegpt4v_d/{}/{}/sharegpt4v_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_sharegpt4v = file.readlines()
                    with open('moellava_d/{}/{}/moellava_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_moellava = file.readlines()

                    attack_name.append(att)
                    dataset_name.append(sett)
                    drop=0            
                    for index, (key) in enumerate(data_BLIP2):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    blip_zhibiao.append(drop)

                    drop=0    
                    for index, (key) in enumerate(data_InstructBLIP):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    InstructBLIP_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_LLaMA_Adapter_v2):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaMA_Adapter_v2_zhibiao.append(drop)
                    drop=0
                    
                    for index, (key) in enumerate(data_LLaVA):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)                            
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaVA_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_MiniGPT):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    MiniGPT_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_mPLUG):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    mPLUG_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_Otter):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    Otter_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_PandaGPT):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    PandaGPT_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_VPGTrans):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    VPGTrans_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_OFv2):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    OFV2_zhibiao.append(drop)

                    

                    drop=0
                    for index, (key) in enumerate(data_xcomposer):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    xcomposer_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_sharegpt4v):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    sharegpt4v_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_moellava):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    moellava_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_LLaVA15):
                        if 1:#
                            pattern = r"-?\d+\.\d+|-?\d+%"
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(6*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaVA15_zhibiao.append(drop)

                else:
                    with open('BLIP2_f/{}_f/{}/BLIP2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_BLIP2 = file.readlines()
                    with open('InstructBLIP_f/{}_f/{}/InstructBLIP_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_InstructBLIP = file.readlines()
                    with open('LLaMA-Adapter-v2_f/{}_f/{}/LLaMA-Adapter-v2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaMA_Adapter_v2 = file.readlines()

                    with open('LLaVA_f/{}_f/{}/LLaVA_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaVA = file.readlines()
                    with open('MiniGPT-4_f/{}_f/{}/MiniGPT-4_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_MiniGPT = file.readlines()
                    with open('mPLUG-Owl_f/{}_f/{}/mPLUG-Owl_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_mPLUG = file.readlines()

                    with open('Otter_f/{}_f/{}/Otter_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_Otter = file.readlines()
                    with open('PandaGPT_f/{}_f/{}/PandaGPT_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_PandaGPT = file.readlines()
                    with open('VPGTrans_f/{}_f/{}/VPGTrans_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_VPGTrans = file.readlines()

                    with open('OFv2_f/{}_f/{}/OFv2_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_OFv2 = file.readlines()
                    with open('internlm-xcomposer_f/{}_f/{}/internlm-xcomposer_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_xcomposer = file.readlines()
                    with open('LLaVA15_f/{}_f/{}/LLaVA15_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_LLaVA15 = file.readlines()

                    

                    with open('sharegpt4v_f/{}_f/{}/sharegpt4v_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_sharegpt4v = file.readlines()
                    
                    with open('moellava_f/{}_f/{}/moellava_{}_gen_len_30_0_shot.txt'.format(ren,sett,att), 'r') as file:
                        data_moellava = file.readlines()



                    attack_name.append(att)
                    dataset_name.append(sett)
                    drop=0            
                    for index, (key) in enumerate(data_BLIP2):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            # 
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    blip_zhibiao.append(drop)

                    drop=0    
                    for index, (key) in enumerate(data_InstructBLIP):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    InstructBLIP_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_LLaMA_Adapter_v2):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaMA_Adapter_v2_zhibiao.append(drop)

                    drop=0
                    
                    for index, (key) in enumerate(data_LLaVA):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaVA_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_MiniGPT):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    MiniGPT_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_mPLUG):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    mPLUG_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_Otter):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    Otter_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_PandaGPT):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    PandaGPT_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_VPGTrans):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    VPGTrans_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_OFv2):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    OFV2_zhibiao.append(drop)

                    

                    drop=0
                    for index, (key) in enumerate(data_sharegpt4v):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    sharegpt4v_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_moellava):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    moellava_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_xcomposer):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    xcomposer_zhibiao.append(drop)

                    drop=0
                    for index, (key) in enumerate(data_LLaVA15):
                        if index==2 or index==6 or index==10:
                            pattern = r'(-?\d+\.\d+)%'
                            matches = re.findall(pattern, key)
                            if float(matches[0])!=0:
                                drop=drop+float(matches[-1])/(3*float(matches[0]))
                            else:
                                drop="none"
                                break
                    LLaVA15_zhibiao.append(drop)
 

print(len(VPGTrans_zhibiao),len(OFV2_zhibiao),len(xcomposer_zhibiao),len(LLaVA15_zhibiao))
# 创建DataFrame来存储分析结果
analysis_df = pd.DataFrame({
    'dataset_name':dataset_name,
    'attack_name': attack_name,
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
    'sharegpt4v': sharegpt4v_zhibiao,
    'moellava': moellava_zhibiao

})

# 将DataFrame保存为Excel文件
analysis_df.to_excel('text_result240305.xlsx', index=False)