import pandas as pd
import pandas as pd
import json
import pdb
import re

attack_name=[]
content_name=[]
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
GeminiProVision_zhibiao=[]
GPT4V_zhibiao=[]

renwu=['bias']

for ren in renwu:
    with open('BLIP2_bn/{}/result.json'.format(ren), 'r') as file:
        data_BLIP2 = json.load(file)
    with open('InstructBLIP_bn/{}/result.json'.format(ren), 'r') as file:
        data_InstructBLIP = json.load(file)
    with open('LLaMA-Adapter-v2_bn/{}/result.json'.format(ren), 'r') as file:
        data_LLaMA_Adapter_v2 = json.load(file)

    with open('LLaVA_bn/{}/result.json'.format(ren), 'r') as file:
        data_LLaVA = json.load(file)
    with open('MiniGPT-4_bn/{}/result.json'.format(ren), 'r') as file:
        data_MiniGPT = json.load(file)
    with open('mPLUG-Owl_bn/{}/result.json'.format(ren), 'r') as file:
        data_mPLUG = json.load(file)

    with open('Otter_bn/{}/result.json'.format(ren), 'r') as file:
        data_Otter = json.load(file)
    with open('PandaGPT_bn/{}/result.json'.format(ren), 'r') as file:
        data_PandaGPT = json.load(file)
    with open('VPGTrans_bn/{}/result.json'.format(ren), 'r') as file:
        data_VPGTrans = json.load(file)

    with open('OFv2_bn/{}/result.json'.format(ren), 'r') as file:
        data_OFv2 = json.load(file)
    with open('internlm-xcomposer_bn/{}/result.json'.format(ren), 'r') as file:
        data_xcomposer = json.load(file)

    with open('LLaVA15_bn/{}/result.json'.format(ren), 'r') as file:
        data_LLaVA15 = json.load(file)

    with open('sharegpt4v_bn/{}/result.json'.format(ren), 'r') as file:
        data_sharegpt4v = json.load(file)

    with open('moellava_bn/{}/result.json'.format(ren), 'r') as file:
        data_moellava = json.load(file)

    with open('GeminiProVision_bn/{}/result.json'.format(ren), 'r') as file:
        data_GeminiProVision = json.load(file)
    with open('GPT4V_bn/{}/result.json'.format(ren), 'r') as file:
        data_GPT4V = json.load(file)
   

    with open('BLIP2_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_BLIP2 = json.load(file)
    with open('InstructBLIP_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_InstructBLIP = json.load(file)
    with open('LLaMA-Adapter-v2_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_LLaMA_Adapter_v2 = json.load(file)

    with open('LLaVA_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_LLaVA = json.load(file)
    with open('MiniGPT-4_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_MiniGPT = json.load(file)
    with open('mPLUG-Owl_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_mPLUG = json.load(file)

    with open('Otter_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_Otter = json.load(file)
    with open('PandaGPT_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_PandaGPT = json.load(file)
    with open('VPGTrans_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_VPGTrans = json.load(file)

    with open('OFv2_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_OFv2 = json.load(file)
    with open('internlm-xcomposer_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_xcomposer = json.load(file)

    with open('LLaVA15_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_LLaVA15 = json.load(file)

    with open('sharegpt4v_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_sharegpt4v = json.load(file)

    with open('moellava_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_moellava = json.load(file)

    
    with open('GeminiProVision_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_GeminiProVision = json.load(file)
    with open('GPT4V_bn/{}_woman/result.json'.format(ren), 'r') as file:
        woman_data_GPT4V = json.load(file)
    

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_BLIP2.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            blip_zhibiao.append(value)
            attack_name.append('Unsafe')
        
            content_name.append(key)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_BLIP2.items()):
        gender=gender+value
    
    blip_zhibiao.append(culture/8)
    attack_name.append('Culture')
    content_name.append('Culture')

    blip_zhibiao.append(black/5)
    attack_name.append('Race')
    content_name.append('black')

    blip_zhibiao.append(white/5)
    attack_name.append('Race')
    content_name.append('white')

    blip_zhibiao.append(yellow/5)
    attack_name.append('Race')
    content_name.append('yellow')

    blip_zhibiao.append(gender/20)
    attack_name.append('Gender')
    content_name.append('gender')


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_InstructBLIP.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            InstructBLIP_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_InstructBLIP.items()):
        gender=gender+value
    
    InstructBLIP_zhibiao.append(culture/8)
    InstructBLIP_zhibiao.append(black/5)
    InstructBLIP_zhibiao.append(white/5)
    InstructBLIP_zhibiao.append(yellow/5)
    InstructBLIP_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_LLaMA_Adapter_v2.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            LLaMA_Adapter_v2_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_LLaMA_Adapter_v2.items()):
        gender=gender+value
    
    LLaMA_Adapter_v2_zhibiao.append(culture/8)
    LLaMA_Adapter_v2_zhibiao.append(black/5)
    LLaMA_Adapter_v2_zhibiao.append(white/5)
    LLaMA_Adapter_v2_zhibiao.append(yellow/5)
    LLaMA_Adapter_v2_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_LLaVA.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            LLaVA_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_LLaVA.items()):
        gender=gender+value
    
    LLaVA_zhibiao.append(culture/8)
    LLaVA_zhibiao.append(black/5)
    LLaVA_zhibiao.append(white/5)
    LLaVA_zhibiao.append(yellow/5)
    LLaVA_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_MiniGPT.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            MiniGPT_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_MiniGPT.items()):
        gender=gender+value
    
    MiniGPT_zhibiao.append(culture/8)
    MiniGPT_zhibiao.append(black/5)
    MiniGPT_zhibiao.append(white/5)
    MiniGPT_zhibiao.append(yellow/5)
    MiniGPT_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_mPLUG.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            mPLUG_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_mPLUG.items()):
        gender=gender+value
    
    mPLUG_zhibiao.append(culture/8)
    mPLUG_zhibiao.append(black/5)
    mPLUG_zhibiao.append(white/5)
    mPLUG_zhibiao.append(yellow/5)
    mPLUG_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_Otter.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            Otter_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_Otter.items()):
        gender=gender+value
    
    Otter_zhibiao.append(culture/8)
    Otter_zhibiao.append(black/5)
    Otter_zhibiao.append(white/5)
    Otter_zhibiao.append(yellow/5)
    Otter_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_PandaGPT.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            PandaGPT_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_PandaGPT.items()):
        gender=gender+value
    
    PandaGPT_zhibiao.append(culture/8)
    PandaGPT_zhibiao.append(black/5)
    PandaGPT_zhibiao.append(white/5)
    PandaGPT_zhibiao.append(yellow/5)
    PandaGPT_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_VPGTrans.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            VPGTrans_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_VPGTrans.items()):
        gender=gender+value
    
    VPGTrans_zhibiao.append(culture/8)
    VPGTrans_zhibiao.append(black/5)
    VPGTrans_zhibiao.append(white/5)
    VPGTrans_zhibiao.append(yellow/5)
    VPGTrans_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_OFv2.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            OFV2_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_OFv2.items()):
        gender=gender+value
    
    OFV2_zhibiao.append(culture/8)
    OFV2_zhibiao.append(black/5)
    OFV2_zhibiao.append(white/5)
    OFV2_zhibiao.append(yellow/5)
    OFV2_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_xcomposer.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            xcomposer_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_xcomposer.items()):
        gender=gender+value
    
    xcomposer_zhibiao.append(culture/8)
    xcomposer_zhibiao.append(black/5)
    xcomposer_zhibiao.append(white/5)
    xcomposer_zhibiao.append(yellow/5)
    xcomposer_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_LLaVA15.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            LLaVA15_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_LLaVA15.items()):
        gender=gender+value
    
    LLaVA15_zhibiao.append(culture/8)
    LLaVA15_zhibiao.append(black/5)
    LLaVA15_zhibiao.append(white/5)
    LLaVA15_zhibiao.append(yellow/5)
    LLaVA15_zhibiao.append(gender/20)

    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_sharegpt4v.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            sharegpt4v_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_sharegpt4v.items()):
        gender=gender+value
    
    sharegpt4v_zhibiao.append(culture/8)
    sharegpt4v_zhibiao.append(black/5)
    sharegpt4v_zhibiao.append(white/5)
    sharegpt4v_zhibiao.append(yellow/5)
    sharegpt4v_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_moellava.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            moellava_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_moellava.items()):
        gender=gender+value
    
    moellava_zhibiao.append(culture/8)
    moellava_zhibiao.append(black/5)
    moellava_zhibiao.append(white/5)
    moellava_zhibiao.append(yellow/5)
    moellava_zhibiao.append(gender/20)


    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_GPT4V.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            GPT4V_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_GPT4V.items()):
        gender=gender+value
    
    GPT4V_zhibiao.append(culture/8)
    GPT4V_zhibiao.append(black/5)
    GPT4V_zhibiao.append(white/5)
    GPT4V_zhibiao.append(yellow/5)
    GPT4V_zhibiao.append(gender/20)



    culture=0
    gender=0
    black=0
    white=0
    yellow=0     
    for index, (key, value) in enumerate(data_GeminiProVision.items()):
        if index>=0 and index<8:
            culture=culture+value
        if index>=8 and index<18:
            gender=gender+value
        if index>=18 and index<24:
            GeminiProVision_zhibiao.append(value)
        if index>=24 and index<29:
            black=black+value
        if index>=29 and index<34:
            white=white+value
        if index>=34 and index<39:
            yellow=yellow+value
    
    for index, (key, value) in enumerate(woman_data_GeminiProVision.items()):
        gender=gender+value
    
    GeminiProVision_zhibiao.append(culture/8)
    GeminiProVision_zhibiao.append(black/5)
    GeminiProVision_zhibiao.append(white/5)
    GeminiProVision_zhibiao.append(yellow/5)
    GeminiProVision_zhibiao.append(gender/20)



analysis_df = pd.DataFrame({
    'attack_name': attack_name,
    'content_name': content_name,    
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
    'sharegpt4v':sharegpt4v_zhibiao,
    'moellava':moellava_zhibiao,
    'GeminiProVision': GeminiProVision_zhibiao,
    'GPT4V': GPT4V_zhibiao


})

# 将DataFrame保存为Excel文件
analysis_df.to_excel('bias240305.xlsx', index=False)