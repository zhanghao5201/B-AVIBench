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

    youxiao_num_BLIP2=0 #1
    heji_BLIP2=0

    youxiao_num_InstructBLIP=0 #2
    heji_InstructBLIP=0

    youxiao_num_LLaAdapterv2=0 #3
    heji_LLaAdapterv2=0

    youxiao_num_LLaVA=0 #4
    heji_LLaVA=0

    youxiao_num_MiniGPT4=0 #5
    heji_MiniGPT=0

    youxiao_num_mPLUG=0 #6
    heji_mPLUG=0

    youxiao_num_Otter=0 #7
    heji_Otter=0

    youxiao_num_PandaGPT=0 #8
    heji_PandaGPT=0

    youxiao_num_VPGTrans=0 #9
    heji_VPGTrans=0

    youxiao_num_OFv2=0 #10
    heji_OFv2=0

    youxiao_num_xcomposer=0 #11
    heji_xcomposer=0


    youxiao_num_LLaVA15=0 #12
    heji_LLaVA15=0    

    youxiao_num_share=0 #13
    heji_share=0

    youxiao_num_moe=0 #14
    heji_moe=0


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

        mean3=0
        mean5=0 
            
        for index, (key, value) in enumerate(data_BLIP2.items()):           

            level=int(key.split("_")[-1])
            # print(level)
            if ren == 'CLS':                           
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_BLIP2=youxiao_num_BLIP2+1
                        heji_BLIP2=heji_BLIP2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_BLIP2=youxiao_num_BLIP2+0
                        heji_BLIP2=heji_BLIP2+0
                         
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))

                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:                            
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_BLIP2=youxiao_num_BLIP2+1
                        heji_BLIP2=heji_BLIP2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_BLIP2=youxiao_num_BLIP2+0
                        heji_BLIP2=heji_BLIP2+0                
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_BLIP2=youxiao_num_BLIP2+1
                        heji_BLIP2=heji_BLIP2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_BLIP2=youxiao_num_BLIP2+0
                        heji_BLIP2=heji_BLIP2+0                
            if level==5:   
                renwu_name.append(ren)
                dataset_name.append(key.split("severity_")[0][:-1])
                corrutipn_name.append(key.split("severity_")[1][:-2])                
                corrutipn_level.append("3-mean")
                
            if level==5:
                # print("k6",mean3)
                if isinstance(mean3, str):
                    blip_zhibiao.append(mean3)                    
                else:
                    blip_zhibiao.append(mean3/3)                    
                mean3=0
                mean5=0 
                

        ###22
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_InstructBLIP.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+1
                        heji_InstructBLIP=heji_InstructBLIP+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+0
                        heji_InstructBLIP=heji_InstructBLIP+0
               
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+1
                        heji_InstructBLIP=heji_InstructBLIP+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+0
                        heji_InstructBLIP=heji_InstructBLIP+0
                
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+1
                        heji_InstructBLIP=heji_InstructBLIP+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_InstructBLIP=youxiao_num_InstructBLIP+0
                        heji_InstructBLIP=heji_InstructBLIP+0
            if level==5:
                if isinstance(mean3, str):
                    InstructBLIP_zhibiao.append(mean3)
                else:
                    InstructBLIP_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ####33    
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_LLaMA_Adapter_v2.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+1
                        heji_LLaAdapterv2=heji_LLaAdapterv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+0
                        heji_LLaAdapterv2=heji_LLaAdapterv2+0   
                
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+1
                        heji_LLaAdapterv2=heji_LLaAdapterv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+0
                        heji_LLaAdapterv2=heji_LLaAdapterv2+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+1
                        heji_LLaAdapterv2=heji_LLaAdapterv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaAdapterv2=youxiao_num_LLaAdapterv2+0
                        heji_LLaAdapterv2=heji_LLaAdapterv2+0
            if level==5:
                if isinstance(mean3, str):
                    LLaMA_Adapter_v2_zhibiao.append(mean3)
                else:
                    LLaMA_Adapter_v2_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ####4444
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_LLaVA.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_LLaVA=youxiao_num_LLaVA+1
                        heji_LLaVA=heji_LLaVA+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA=youxiao_num_LLaVA+0
                        heji_LLaVA=heji_LLaVA+0  
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_LLaVA=youxiao_num_LLaVA+1
                        heji_LLaVA=heji_LLaVA+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA=youxiao_num_LLaVA+0
                        heji_LLaVA=heji_LLaVA+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_LLaVA=youxiao_num_LLaVA+1
                        heji_LLaVA=heji_LLaVA+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA=youxiao_num_LLaVA+0
                        heji_LLaVA=heji_LLaVA+0
            if level==5:
                if isinstance(mean3, str):
                    LLaVA_zhibiao.append(mean3)
                else:
                    LLaVA_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ####555
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_MiniGPT.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+1
                        heji_MiniGPT=heji_MiniGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+0
                        heji_MiniGPT=heji_MiniGPT+0 
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+1
                        heji_MiniGPT=heji_MiniGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+0
                        heji_MiniGPT=heji_MiniGPT+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+1
                        heji_MiniGPT=heji_MiniGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_MiniGPT4=youxiao_num_MiniGPT4+0
                        heji_MiniGPT=heji_MiniGPT+0
            if level==5:
                if isinstance(mean3, str):
                    MiniGPT_zhibiao.append(mean3)
                else:
                    MiniGPT_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ####666
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_mPLUG.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_mPLUG=youxiao_num_mPLUG+1
                        heji_mPLUG=heji_mPLUG+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_mPLUG=youxiao_num_mPLUG+0
                        heji_mPLUG=heji_mPLUG+0  
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_mPLUG=youxiao_num_mPLUG+1
                        heji_mPLUG=heji_mPLUG+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_mPLUG=youxiao_num_mPLUG+0
                        heji_mPLUG=heji_mPLUG+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_mPLUG=youxiao_num_mPLUG+1
                        heji_mPLUG=heji_mPLUG+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_mPLUG=youxiao_num_mPLUG+0
                        heji_mPLUG=heji_mPLUG+0
            if level==5:
                if isinstance(mean3, str):
                    mPLUG_zhibiao.append(mean3)
                else:
                    mPLUG_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ###777
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_Otter.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_Otter=youxiao_num_Otter+1
                        heji_Otter=heji_Otter+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_Otter=youxiao_num_Otter+0
                        heji_Otter=heji_Otter+0
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_Otter=youxiao_num_Otter+1
                        heji_Otter=heji_Otter+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_Otter=youxiao_num_Otter+0
                        heji_Otter=heji_Otter+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_Otter=youxiao_num_Otter+1
                        heji_Otter=heji_Otter+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_Otter=youxiao_num_Otter+0
                        heji_Otter=heji_Otter+0
            if level==5:
                if isinstance(mean3, str):
                    Otter_zhibiao.append(mean3)
                else:
                    Otter_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 
        ###888
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_PandaGPT.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+1
                        heji_PandaGPT=heji_PandaGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+0
                        heji_PandaGPT=heji_PandaGPT+0  
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+1
                        heji_PandaGPT=heji_PandaGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+0
                        heji_PandaGPT=heji_PandaGPT+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+1
                        heji_PandaGPT=heji_PandaGPT+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_PandaGPT=youxiao_num_PandaGPT+0
                        heji_PandaGPT=heji_PandaGPT+0
            if level==5:
                if isinstance(mean3, str):
                    PandaGPT_zhibiao.append(mean3)
                else:
                    PandaGPT_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        ###999
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_VPGTrans.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+1
                        heji_VPGTrans=heji_VPGTrans+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+0
                        heji_VPGTrans=heji_VPGTrans+0   
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+1
                        heji_VPGTrans=heji_VPGTrans+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+0
                        heji_VPGTrans=heji_VPGTrans+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+1
                        heji_VPGTrans=heji_VPGTrans+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_VPGTrans=youxiao_num_VPGTrans+0
                        heji_VPGTrans=heji_VPGTrans+0
            if level==5:
                if isinstance(mean3, str):
                    VPGTrans_zhibiao.append(mean3)
                else:
                    VPGTrans_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 

        #OFV2_zhibiao
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_OFv2.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_OFv2=youxiao_num_OFv2+1
                        heji_OFv2=heji_OFv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_OFv2=youxiao_num_OFv2+0
                        heji_OFv2=heji_OFv2+0  
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_OFv2=youxiao_num_OFv2+1
                        heji_OFv2=heji_OFv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_OFv2=youxiao_num_OFv2+0
                        heji_OFv2=heji_OFv2+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_OFv2=youxiao_num_OFv2+1
                        heji_OFv2=heji_OFv2+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_OFv2=youxiao_num_OFv2+0
                        heji_OFv2=heji_OFv2+0
            if level==5:
                if isinstance(mean3, str):
                    OFV2_zhibiao.append(mean3)
                else:
                    OFV2_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_xcomposer.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_xcomposer=youxiao_num_xcomposer+1
                        heji_xcomposer=heji_xcomposer+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_xcomposer=youxiao_num_xcomposer+0
                        heji_xcomposer=heji_xcomposer+0  
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_xcomposer=youxiao_num_xcomposer+1
                        heji_xcomposer=heji_xcomposer+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_xcomposer=youxiao_num_xcomposer+0
                        heji_xcomposer=heji_xcomposer+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            # print(mean3)
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_xcomposer=youxiao_num_xcomposer+1
                        heji_xcomposer=heji_xcomposer+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_xcomposer=youxiao_num_xcomposer+0
                        heji_xcomposer=heji_xcomposer+0
            if level==5:
                if isinstance(mean3, str):
                    xcomposer_zhibiao.append(mean3)
                else:
                    xcomposer_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 


        #LLaVA15_ff
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_LLaVA15.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+1
                        heji_LLaVA15=heji_LLaVA15+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+0
                        heji_LLaVA15=heji_LLaVA15+0   
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+1
                        heji_LLaVA15=heji_LLaVA15+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+0
                        heji_LLaVA15=heji_LLaVA15+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+1
                        heji_LLaVA15=heji_LLaVA15+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_LLaVA15=youxiao_num_LLaVA15+0
                        heji_LLaVA15=heji_LLaVA15+0
            if level==5:
                if isinstance(mean3, str):
                    LLaVA15_zhibiao.append(mean3)
                else:
                    LLaVA15_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 
    

        #moe
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_moe.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_moe=youxiao_num_moe+1
                        heji_moe=heji_moe+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_moe=youxiao_num_moe+0
                        heji_moe=heji_moe+0   
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_moe=youxiao_num_moe+1
                        heji_moe=heji_moe+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_moe=youxiao_num_moe+0
                        heji_moe=heji_moe+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            # print(mean3)
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_moe=youxiao_num_moe+1
                        heji_moe=heji_moe+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_moe=youxiao_num_moe+0
                        heji_moe=heji_moe+0
            if level==5:
                if isinstance(mean3, str):
                    moe_zhibiao.append(mean3)
                else:
                    moe_zhibiao.append(mean3/3)
                mean3=0
                mean5=0 


        #share
        mean3=0
        mean5=0  
        for index, (key, value) in enumerate(data_share.items()):
            level=int(key.split("_")[-1])
            if ren == 'CLS':
                if level==0:
                    init = value['mean_per_class_acc']
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value['mean_per_class_acc'])/init
                            mean5+= (init-value['mean_per_class_acc'])/init
                        else:
                            mean5+= (init-value['mean_per_class_acc'])/init 
                        youxiao_num_share=youxiao_num_share+1
                        heji_share=heji_share+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_share=youxiao_num_share+0
                        heji_share=heji_share+0 
            elif ren=="KIE":
                f1_value = float(re.search(r"F1: (\d+\.\d+)", value).group(1))
                if level==0:
                    init = f1_value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-f1_value)/init
                            mean5+= (init-f1_value)/init
                        else:
                            mean5+= (init-f1_value)/init
                        youxiao_num_share=youxiao_num_share+1
                        heji_share=heji_share+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_share=youxiao_num_share+0
                        heji_share=heji_share+0
            else:
                if level==0:
                    init = value
                else:
                    if init!=0:
                        if level==3 or level==5 or level==1:
                            mean3+= (init-value)/init
                            mean5+= (init-value)/init
                        else:
                            mean5+= (init-value)/init
                        youxiao_num_share=youxiao_num_share+1
                        heji_share=heji_share+mean3
                    else:
                        mean3='none'
                        mean5='none'
                        youxiao_num_share=youxiao_num_share+0
                        heji_share=heji_share+0
            if level==5:
                if isinstance(mean3, str):
                    share_zhibiao.append(mean3)
                else:
                    share_zhibiao.append(mean3/3)
                mean3=0
                mean5=0  

    renwu_name.append(neng+"num")
    dataset_name.append(neng+"num")
    corrutipn_name.append(neng+"num")
    corrutipn_level.append(neng+"num")
    renwu_name.append(neng+'shuju')
    dataset_name.append(neng+'shuju')
    corrutipn_name.append(neng+'shuju')
    corrutipn_level.append(neng+'shuju')

    blip_zhibiao.append(youxiao_num_BLIP2)
    InstructBLIP_zhibiao.append(youxiao_num_InstructBLIP)
    LLaMA_Adapter_v2_zhibiao.append(youxiao_num_LLaAdapterv2)
    LLaVA_zhibiao.append(youxiao_num_LLaVA)
    MiniGPT_zhibiao.append(youxiao_num_MiniGPT4)
    mPLUG_zhibiao.append(youxiao_num_mPLUG)
    Otter_zhibiao.append(youxiao_num_Otter)
    PandaGPT_zhibiao.append(youxiao_num_PandaGPT)
    VPGTrans_zhibiao.append(youxiao_num_VPGTrans)
    OFV2_zhibiao.append(youxiao_num_OFv2)
    xcomposer_zhibiao.append(youxiao_num_xcomposer)
    LLaVA15_zhibiao.append(youxiao_num_LLaVA15)

    moe_zhibiao.append(youxiao_num_moe)
    share_zhibiao.append(youxiao_num_share)

    blip_zhibiao.append(heji_BLIP2/youxiao_num_BLIP2)
    InstructBLIP_zhibiao.append(heji_InstructBLIP/youxiao_num_InstructBLIP)
    LLaMA_Adapter_v2_zhibiao.append(heji_LLaAdapterv2/youxiao_num_LLaAdapterv2)
    LLaVA_zhibiao.append(heji_LLaVA/youxiao_num_LLaVA)
    MiniGPT_zhibiao.append(heji_MiniGPT/youxiao_num_MiniGPT4)
    mPLUG_zhibiao.append(heji_mPLUG/youxiao_num_mPLUG)
    Otter_zhibiao.append(heji_Otter/youxiao_num_Otter)
    PandaGPT_zhibiao.append(heji_PandaGPT/youxiao_num_PandaGPT)
    VPGTrans_zhibiao.append(heji_VPGTrans/youxiao_num_VPGTrans)
    OFV2_zhibiao.append(heji_OFv2/youxiao_num_OFv2)
    xcomposer_zhibiao.append(heji_xcomposer/youxiao_num_xcomposer)
    LLaVA15_zhibiao.append(heji_LLaVA15/youxiao_num_LLaVA15)

    moe_zhibiao.append(heji_moe/youxiao_num_moe)
    share_zhibiao.append(heji_share/youxiao_num_share)



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
analysis_df.to_excel('corruption_result240305.xlsx', index=False)