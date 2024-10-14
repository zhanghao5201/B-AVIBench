import pandas as pd
import pandas as pd
import json
import pdb
import re

Capability_name=[]
renwu_name=[]
dataset_name=[]
Model_name=[]
ceshi_name=[] 

data={}

zhibiao={}

Model=['BLIP2_f','InstructBLIP_f','LLaMA-Adapter-v2_f','LLaVA_f','MiniGPT-4_f','mPLUG-Owl_f','Otter_f','PandaGPT_f','VPGTrans_f','OFv2_f','internlm-xcomposer_f','LLaVA15_f','sharegpt4v_f' , 'moellava_f']#,'internlm-xcomposer','LLaVA15']


renwu=['mci','cls','ocr','kie','vqa','vqachoice','object','imagenetvc']
Ceshi=['success_rate','attack_noise','attack_patch','attack_patch_boundary','attack_patch_SurFree']
a_num = 'attack_num'
dataset=['MSCOCO_OC','VCR1_OC','OC_mean','MSCOCO_MCI','VCR1_MCI','MCI_mean','CIFAR10','CIFAR100','OxfordIIITPet','Flowers102','ImageNet','cls_mean','Vis_p_mean',
'COCO-Text','CTW','CUTE80','HOST','IC13','IC15','IIIT5K','SVTP','SVT','Total-Text','WOST','WordArt','ocr_mean','FUNSD','POIE','SROIE','kie_mean','Vis_k_mean',
'AOKVQAClose','AOKVQAOpen','DocVQA','GQA','OCRVQA','OKVQA','STVQA','TextVQA','VizWiz','WHOOPSVQA','WHOOPSWeird','Visdial','vqa_mean',
'ScienceQAIMG','IconQA','VSR','vqachoice_mean','Vis_R_mean',
'MSCOCO_pope_random','MSCOCO_pope_adversarial','MSCOCO_pope_popular','pope_mean','OB_H_mean',
'ImageNetVC_color','ImageNetVC_component','ImageNetVC_material','ImageNetVC_others','ImageNetVC_shape','imagenetvc_mean','Vis_C_mean',
]

for model in Model:
    data[model]={}
for nn_d in dataset:
    zhibiao[nn_d]=[]
vis_r_n_jilu={}
vis_r_f_jilu={}

vis_k_n_jilu={}
vis_k_f_jilu={}

vis_p_n_jilu_1={}
vis_p_f_jilu_1={}

vis_p_n_jilu_2={}
vis_p_f_jilu_2={}

for model in Model:
    vis_r_n_jilu[model]={}
    vis_r_f_jilu[model]={}

    vis_k_n_jilu[model]={}
    vis_k_f_jilu[model]={}

    vis_p_n_jilu_1[model]={}
    vis_p_f_jilu_1[model]={}

    vis_p_n_jilu_2[model]={}
    vis_p_f_jilu_2[model]={}

for ren in renwu:
    for model in Model:
        print("kls",model,ren)        
        if ren=='mci':
            with open('{}/{}/result_VCR1_OC.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='cls':
            with open('{}/{}/result_OxfordIIITPet.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='ocr':
            with open('{}/{}/result_WordArt.json'.format(model,ren), 'r') as file:
                text=json.load(file)    
        if ren=='kie':
            with open('{}/{}/result_SROIE.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='vqa':
            with open('{}/{}/result_Visdial.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='vqachoice':    
            with open('{}/{}/result_VSR.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='imagenetvc':
            with open('{}/{}/result_ImageNetVC_shape.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        if ren=='object':
            with open('{}/{}/result_MSCOCO_pope_popular.json'.format(model,ren), 'r') as file:
                text=json.load(file)
        
        for ceshi in Ceshi:
            data[model][ren] = text
            if ren=='mci':
                Model_name.append(model)
                ceshi_name.append(ceshi)
            if ren=='mci':
                if ceshi=='success_rate':
                    a0=data[model][ren]['{}_severity_0'.format(dataset[0])][ceshi]
                    a1=data[model][ren]['{}_severity_0'.format(dataset[1])][ceshi]

                    a2=data[model][ren]['{}_severity_0'.format(dataset[3])][ceshi]
                    a3=data[model][ren]['{}_severity_0'.format(dataset[4])][ceshi]  

                    n0=data[model][ren]['{}_severity_0'.format(dataset[0])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[1])][a_num]

                    n2=data[model][ren]['{}_severity_0'.format(dataset[3])][a_num]
                    n3=data[model][ren]['{}_severity_0'.format(dataset[4])][a_num]  

                    zhibiao[dataset[0]].append(a0)
                    zhibiao[dataset[1]].append(a1)
                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0
                    if a3==0 or a3==-100:
                        a3=0

                    zhibiao[dataset[2]].append((a0*n0+a1*n1)/(n0+n1))
                    vis_p_n_1=[(a0*n0+a1*n1),(n0+n1)]
                    vis_p_n_jilu_1[model]['num']=vis_p_n_1

                    zhibiao[dataset[3]].append(a2)
                    zhibiao[dataset[4]].append(a3)
                    zhibiao[dataset[5]].append((a2*n2+a3*n3)/(n2+n3))
                    vis_p_n_2=[(a2*n2+a3*n3),(n2+n3)]
                    vis_p_n_jilu_2[model]['num']=vis_p_n_2
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[0])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[1])][ceshi]

                    b2=data[model][ren]['{}_severity_0'.format(dataset[3])][ceshi]
                    b3=data[model][ren]['{}_severity_0'.format(dataset[4])][ceshi]

                    zhibiao[dataset[0]].append(b0)
                    zhibiao[dataset[1]].append(b1)
                    if a0==0 and a1==0:
                        zhibiao[dataset[2]].append('fail')
                    else:              
                        zhibiao[dataset[2]].append((b0*a0*n0+b1*a1*n1)/(vis_p_n_1[0]))    
                    vis_p_f_1= [(b0*a0*n0+b1*a1*n1),(vis_p_n_1[0])] 

                    vis_p_f_jilu_1[model][ceshi]=vis_p_f_1

                    zhibiao[dataset[3]].append(b2)
                    zhibiao[dataset[4]].append(b3)
                    if a0==0 and a3==0:
                        zhibiao[dataset[5]].append('fail')
                    else:
                        zhibiao[dataset[5]].append((b2*a2*n2+b3*a3*n3)/(vis_p_n_2[0]))
                    vis_p_f_2= [(b2*a2*n2+b3*a3*n3),(vis_p_n_2[0])]
                    vis_p_f_jilu_2[model][ceshi]=vis_p_f_2 

            elif ren=='cls':
                if ceshi=='success_rate':
                    c0=data[model][ren]['{}_severity_0'.format(dataset[6])][ceshi]
                    c1=data[model][ren]['{}_severity_0'.format(dataset[7])][ceshi]
                    c2=data[model][ren]['{}_severity_0'.format(dataset[8])][ceshi]
                    c3=data[model][ren]['{}_severity_0'.format(dataset[9])][ceshi]
                    c4=data[model][ren]['{}_severity_0'.format(dataset[10])][ceshi]

                    m0=data[model][ren]['{}_severity_0'.format(dataset[6])][a_num]
                    m1=data[model][ren]['{}_severity_0'.format(dataset[7])][a_num]
                    m2=data[model][ren]['{}_severity_0'.format(dataset[8])][a_num]
                    m3=data[model][ren]['{}_severity_0'.format(dataset[9])][a_num]
                    m4=data[model][ren]['{}_severity_0'.format(dataset[10])][a_num]

                    zhibiao[dataset[6]].append(c0)
                    zhibiao[dataset[7]].append(c1)                
                    zhibiao[dataset[8]].append(c2)
                    zhibiao[dataset[9]].append(c3)
                    zhibiao[dataset[10]].append(c4)
                    if c0==0 or c0==-100:
                        c0=0
                    if c1==0 or c1==-100:
                        c1=0
                    if c2==0 or c2==-100:
                        c2=0
                    if c3==0 or c3==-100:
                        c3=0
                    if c4==0 or c4==-100:
                        c4=0

                    zhibiao[dataset[11]].append((c0*m0+c1*m1+c2*m2+c3*m3+c4*m4)/(m0+m1+m2+m3+m4))
                    vis_p_n_3=[(c0*m0+c1*m1+c2*m2+c3*m3+c4*m4),(m0+m1+m2+m3+m4)]
                    zhibiao[dataset[12]].append((vis_p_n_jilu_2[model]['num'][0]+vis_p_n_jilu_1[model]['num'][0]+vis_p_n_3[0])/(vis_p_n_jilu_2[model]['num'][1]+vis_p_n_jilu_1[model]['num'][1]+vis_p_n_3[1]))
                else:
                    d0=data[model][ren]['{}_severity_0'.format(dataset[6])][ceshi]
                    d1=data[model][ren]['{}_severity_0'.format(dataset[7])][ceshi]
                    d2=data[model][ren]['{}_severity_0'.format(dataset[8])][ceshi]
                    d3=data[model][ren]['{}_severity_0'.format(dataset[9])][ceshi]
                    d4=data[model][ren]['{}_severity_0'.format(dataset[10])][ceshi]

                    zhibiao[dataset[6]].append(d0)
                    zhibiao[dataset[7]].append(d1)                
                    zhibiao[dataset[8]].append(d2)
                    zhibiao[dataset[9]].append(d3)
                    zhibiao[dataset[10]].append(d4)
                    zhibiao[dataset[11]].append((c0*d0*m0+c1*d1*m1+c2*d2*m2+c3*d3*m3+c4*d4*m4)/vis_p_n_3[0])
                    vis_p_f_3= [(c0*d0*m0+c1*d1*m1+c2*d2*m2+c3*d3*m3+c4*d4*m4),vis_p_n_3[0]] 
                    vis_p_zi=(vis_p_f_jilu_1[model][ceshi][0]+vis_p_f_jilu_2[model][ceshi][0]+vis_p_f_3[0])/(vis_p_f_jilu_1[model][ceshi][1]+vis_p_f_jilu_2[model][ceshi][1]+vis_p_f_3[1])       
                    zhibiao[dataset[12]].append(vis_p_zi)
            elif ren=='ocr': #12
                if ceshi=='success_rate':
                    a0=data[model][ren]['{}_severity_0'.format(dataset[13])][ceshi]
                    a1=data[model][ren]['{}_severity_0'.format(dataset[14])][ceshi]
                    a2=data[model][ren]['{}_severity_0'.format(dataset[15])][ceshi]
                    a3=data[model][ren]['{}_severity_0'.format(dataset[16])][ceshi]
                    a4=data[model][ren]['{}_severity_0'.format(dataset[17])][ceshi]
                    a5=data[model][ren]['{}_severity_0'.format(dataset[18])][ceshi]

                    a6=data[model][ren]['{}_severity_0'.format(dataset[19])][ceshi]
                    a7=data[model][ren]['{}_severity_0'.format(dataset[20])][ceshi]
                    a8=data[model][ren]['{}_severity_0'.format(dataset[21])][ceshi]
                    a9=data[model][ren]['{}_severity_0'.format(dataset[22])][ceshi]
                    a10=data[model][ren]['{}_severity_0'.format(dataset[23])][ceshi]
                    a11=data[model][ren]['{}_severity_0'.format(dataset[24])][ceshi]

                    n0=data[model][ren]['{}_severity_0'.format(dataset[13])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[14])][a_num]
                    n2=data[model][ren]['{}_severity_0'.format(dataset[15])][a_num]
                    n3=data[model][ren]['{}_severity_0'.format(dataset[16])][a_num]
                    n4=data[model][ren]['{}_severity_0'.format(dataset[17])][a_num]
                    n5=data[model][ren]['{}_severity_0'.format(dataset[18])][a_num]

                    n6=data[model][ren]['{}_severity_0'.format(dataset[19])][a_num]
                    n7=data[model][ren]['{}_severity_0'.format(dataset[20])][a_num]
                    n8=data[model][ren]['{}_severity_0'.format(dataset[21])][a_num]
                    n9=data[model][ren]['{}_severity_0'.format(dataset[22])][a_num]
                    n10=data[model][ren]['{}_severity_0'.format(dataset[23])][a_num]
                    n11=data[model][ren]['{}_severity_0'.format(dataset[24])][a_num]

                    zhibiao[dataset[13]].append(a0)
                    zhibiao[dataset[14]].append(a1) 
                    zhibiao[dataset[15]].append(a2)
                    zhibiao[dataset[16]].append(a3) 
                    zhibiao[dataset[17]].append(a4)
                    zhibiao[dataset[18]].append(a5) 
                    zhibiao[dataset[19]].append(a6)
                    zhibiao[dataset[20]].append(a7) 
                    zhibiao[dataset[21]].append(a8)
                    zhibiao[dataset[22]].append(a9) 
                    zhibiao[dataset[23]].append(a10)
                    zhibiao[dataset[24]].append(a11) 
                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0
                    if a3==0 or a3==-100:
                        a3=0
                    if a4==0 or a4==-100:
                        a4=0
                    if a5==0 or a5==-100:
                        a5=0
                    if a6==0 or a6==-100:
                        a6=0
                    if a7==0 or a7==-100:
                        a7=0
                    if a8==0 or a8==-100:
                        a8=0
                    if a9==0 or a9==-100:
                        a9=0
                    if a10==0 or a10==-100:
                        a10=0
                    if a11==0 or a11==-100:
                        a11=0

                    zhibiao[dataset[25]].append((a0*n0+a1*n1+a2*n2+a3*n3+a4*n4+a5*n5+a6*n6+a7*n7+a8*n8+a9*n9+a10*n10+a11*n11)/(n0+n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11))
                    vis_k_n_1=[(a0*n0+a1*n1+a2*n2+a3*n3+a4*n4+a5*n5+a6*n6+a7*n7+a8*n8+a9*n9+a10*n10+a11*n11),(n0+n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11)]
                    vis_k_n_jilu[model]['num']=vis_k_n_1
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[13])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[14])][ceshi]
                    b2=data[model][ren]['{}_severity_0'.format(dataset[15])][ceshi]
                    b3=data[model][ren]['{}_severity_0'.format(dataset[16])][ceshi]
                    b4=data[model][ren]['{}_severity_0'.format(dataset[17])][ceshi]
                    b5=data[model][ren]['{}_severity_0'.format(dataset[18])][ceshi]

                    b6=data[model][ren]['{}_severity_0'.format(dataset[19])][ceshi]
                    b7=data[model][ren]['{}_severity_0'.format(dataset[20])][ceshi]
                    b8=data[model][ren]['{}_severity_0'.format(dataset[21])][ceshi]
                    b9=data[model][ren]['{}_severity_0'.format(dataset[22])][ceshi]
                    b10=data[model][ren]['{}_severity_0'.format(dataset[23])][ceshi]
                    b11=data[model][ren]['{}_severity_0'.format(dataset[24])][ceshi]

                    zhibiao[dataset[13]].append(b0)
                    zhibiao[dataset[14]].append(b1) 
                    zhibiao[dataset[15]].append(b2)
                    zhibiao[dataset[16]].append(b3) 
                    zhibiao[dataset[17]].append(b4)
                    zhibiao[dataset[18]].append(b5) 
                    zhibiao[dataset[19]].append(b6)
                    zhibiao[dataset[20]].append(b7) 
                    zhibiao[dataset[21]].append(b8)
                    zhibiao[dataset[22]].append(b9) 
                    zhibiao[dataset[23]].append(b10)
                    zhibiao[dataset[24]].append(b11)

                    zhibiao[dataset[25]].append((b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4+b5*a5*n5+b6*a6*n6+b7*a7*n7+b8*a8*n8+b9*a9*n9+b10*a10*n10+b11*a11*n11)/vis_k_n_1[0])    
                    vis_k_f_1= [(b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4+b5*a5*n5+b6*a6*n6+b7*a7*n7+b8*a8*n8+b9*a9*n9+b10*a10*n10+b11*a11*n11),vis_k_n_1[0]] 
                    vis_k_f_jilu[model][ceshi]=vis_k_f_1
            elif ren=='kie':
                if ceshi=='success_rate':
                    a0=data[model][ren]['{}_severity_0'.format(dataset[26])][ceshi]
                    a1=data[model][ren]['{}_severity_0'.format(dataset[27])][ceshi]
                    a2=data[model][ren]['{}_severity_0'.format(dataset[28])][ceshi]

                    n0=data[model][ren]['{}_severity_0'.format(dataset[26])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[27])][a_num]
                    n2=data[model][ren]['{}_severity_0'.format(dataset[28])][a_num]

                    zhibiao[dataset[26]].append(a0)
                    zhibiao[dataset[27]].append(a1) 
                    zhibiao[dataset[28]].append(a2)

                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0

                    zhibiao[dataset[29]].append((a0*n0+a1*n1+a2*n2)/(n1+n2+n0))
                    vis_k_n_2=[(a0*n0+a1*n1+a2*n2),(n1+n2+n0)]
                    zhibiao[dataset[30]].append((vis_k_n_2[0]+vis_k_n_jilu[model]['num'][0])/(vis_k_n_2[1]+vis_k_n_jilu[model]['num'][1]))
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[26])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[27])][ceshi]
                    b2=data[model][ren]['{}_severity_0'.format(dataset[28])][ceshi]

                    zhibiao[dataset[26]].append(b0)
                    zhibiao[dataset[27]].append(b1) 
                    zhibiao[dataset[28]].append(b2)

                    zhibiao[dataset[29]].append((b0*a0*n0+b1*a1*n1+b2*a2*n2)/(vis_k_n_2[0]))    
                    vis_k_f_2= [(b0*a0*n0+b1*a1*n1+b2*a2*n2),(vis_k_n_2[0])]
                    vis_k_zi=(vis_k_f_jilu[model][ceshi][0]+vis_k_f_2[0])/(vis_k_f_jilu[model][ceshi][1]+vis_k_f_2[1])                     
                    zhibiao[dataset[30]].append(vis_k_zi) 
            elif ren=='vqa':
                if ceshi=='success_rate':
                    n0=data[model][ren]['{}_severity_0'.format(dataset[31])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[32])][a_num]
                    n2=data[model][ren]['{}_severity_0'.format(dataset[33])][a_num]
                    n3=data[model][ren]['{}_severity_0'.format(dataset[34])][a_num]
                    n4=data[model][ren]['{}_severity_0'.format(dataset[35])][a_num]
                    n5=data[model][ren]['{}_severity_0'.format(dataset[36])][a_num]

                    n6=data[model][ren]['{}_severity_0'.format(dataset[37])][a_num]
                    n7=data[model][ren]['{}_severity_0'.format(dataset[38])][a_num]
                    n8=data[model][ren]['{}_severity_0'.format(dataset[39])][a_num]
                    n9=data[model][ren]['{}_severity_0'.format(dataset[40])][a_num]
                    n10=data[model][ren]['{}_severity_0'.format(dataset[41])][a_num]
                    n11=data[model][ren]['{}_severity_0'.format(dataset[42])][a_num]
                    
                    a0=data[model][ren]['{}_severity_0'.format(dataset[31])][ceshi]   
                    a1=data[model][ren]['{}_severity_0'.format(dataset[32])][ceshi]
                    a2=data[model][ren]['{}_severity_0'.format(dataset[33])][ceshi]
                    a3=data[model][ren]['{}_severity_0'.format(dataset[34])][ceshi]
                    a4=data[model][ren]['{}_severity_0'.format(dataset[35])][ceshi]
                    a5=data[model][ren]['{}_severity_0'.format(dataset[36])][ceshi]

                    a6=data[model][ren]['{}_severity_0'.format(dataset[37])][ceshi]
                    a7=data[model][ren]['{}_severity_0'.format(dataset[38])][ceshi]
                    a8=data[model][ren]['{}_severity_0'.format(dataset[39])][ceshi]
                    a9=data[model][ren]['{}_severity_0'.format(dataset[40])][ceshi]
                    a10=data[model][ren]['{}_severity_0'.format(dataset[41])][ceshi]
                    a11=data[model][ren]['{}_severity_0'.format(dataset[42])][ceshi]                

                    zhibiao[dataset[31]].append(a0)
                    zhibiao[dataset[32]].append(a1) 
                    zhibiao[dataset[33]].append(a2)
                    zhibiao[dataset[34]].append(a3) 
                    zhibiao[dataset[35]].append(a4)
                    zhibiao[dataset[36]].append(a5) 
                    zhibiao[dataset[37]].append(a6)
                    zhibiao[dataset[38]].append(a7) 
                    zhibiao[dataset[39]].append(a8)
                    zhibiao[dataset[40]].append(a9) 
                    zhibiao[dataset[41]].append(a10)
                    zhibiao[dataset[42]].append(a11) 

                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0
                    if a3==0 or a3==-100:
                        a3=0
                    if a4==0 or a4==-100:
                        a4=0
                    if a5==0 or a5==-100:
                        a5=0
                    if a6==0 or a6==-100:
                        a6=0
                    if a7==0 or a7==-100:
                        a7=0
                    if a8==0 or a8==-100:
                        a8=0
                    if a9==0 or a9==-100:
                        a9=0
                    if a10==0 or a10==-100:
                        a10=0
                    if a11==0 or a11==-100:
                        a11=0
                    zhibiao[dataset[43]].append((a0*n0+a1*n1+a2*n2+a3*n3+a4*n4+a5*n5+a6*n6+a7*n7+a8*n8+a9*n9+a10*n10+a11*n11)/(n0+n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11))
                    vis_r_n_1=[(a0*n0+a1*n1+a2*n2+a3*n3+a4*n4+a5*n5+a6*n6+a7*n7+a8*n8+a9*n9+a10*n10+a11*n11),(n0+n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11)]
                    vis_r_n_jilu[model]['num']=vis_r_n_1#                    
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[31])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[32])][ceshi]
                    b2=data[model][ren]['{}_severity_0'.format(dataset[33])][ceshi]
                    b3=data[model][ren]['{}_severity_0'.format(dataset[34])][ceshi]
                    b4=data[model][ren]['{}_severity_0'.format(dataset[35])][ceshi]
                    b5=data[model][ren]['{}_severity_0'.format(dataset[36])][ceshi]

                    b6=data[model][ren]['{}_severity_0'.format(dataset[37])][ceshi]
                    b7=data[model][ren]['{}_severity_0'.format(dataset[38])][ceshi]
                    b8=data[model][ren]['{}_severity_0'.format(dataset[39])][ceshi]
                    b9=data[model][ren]['{}_severity_0'.format(dataset[40])][ceshi]
                    b10=data[model][ren]['{}_severity_0'.format(dataset[41])][ceshi]
                    b11=data[model][ren]['{}_severity_0'.format(dataset[42])][ceshi]
                    zhibiao[dataset[31]].append(b0)
                    zhibiao[dataset[32]].append(b1) 
                    zhibiao[dataset[33]].append(b2)
                    zhibiao[dataset[34]].append(b3) 
                    zhibiao[dataset[35]].append(b4)
                    zhibiao[dataset[36]].append(b5) 
                    zhibiao[dataset[37]].append(b6)
                    zhibiao[dataset[38]].append(b7) 
                    zhibiao[dataset[39]].append(b8)
                    zhibiao[dataset[40]].append(b9) 
                    zhibiao[dataset[41]].append(b10)
                    zhibiao[dataset[42]].append(b11)
                    zhibiao[dataset[43]].append((b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4+b5*a5*n5+b6*a6*n6+b7*a7*n7+b8*a8*n8+b9*a9*n9+b10*a10*n10+b11*a11*n11)/vis_r_n_1[0])    
                    # print("opd",vis_r_n_1[0])
                    vis_r_f_1= [(b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4+b5*a5*n5+b6*a6*n6+b7*a7*n7+b8*a8*n8+b9*a9*n9+b10*a10*n10+b11*a11*n11),vis_r_n_1[0]] 
                    vis_r_f_jilu[model][ceshi]=vis_r_f_1
            elif ren=='vqachoice':
                if ceshi=='success_rate':
                    c0=data[model][ren]['{}_severity_0'.format(dataset[44])][ceshi]
                    c1=data[model][ren]['{}_severity_0'.format(dataset[45])][ceshi]
                    c2=data[model][ren]['{}_severity_0'.format(dataset[46])][ceshi]

                    m0=data[model][ren]['{}_severity_0'.format(dataset[44])][a_num]
                    m1=data[model][ren]['{}_severity_0'.format(dataset[45])][a_num]
                    m2=data[model][ren]['{}_severity_0'.format(dataset[46])][a_num]

                    zhibiao[dataset[44]].append(c0)
                    zhibiao[dataset[45]].append(c1) 
                    zhibiao[dataset[46]].append(c2)

                    if c0==0 or c0==-100:
                        c0=0
                    if c1==0 or c1==-100:
                        c1=0
                    if c2==0 or c2==-100:
                        c2=0

                    zhibiao[dataset[47]].append((c0*m0+c1*m1+c2*m2)/(m0+m1+m2))
                    vis_r_n_2=[(c0*m0+c1*m1+c2*m2),(m0+m1+m2)]

                    zhibiao[dataset[48]].append((vis_r_n_2[0]+vis_r_n_jilu[model]['num'][0])/(vis_r_n_2[1]+vis_r_n_jilu[model]['num'][1]))
                else:
                    d0=data[model][ren]['{}_severity_0'.format(dataset[44])][ceshi]
                    d1=data[model][ren]['{}_severity_0'.format(dataset[45])][ceshi]
                    d2=data[model][ren]['{}_severity_0'.format(dataset[46])][ceshi]
                   

                    zhibiao[dataset[44]].append(d0)
                    zhibiao[dataset[45]].append(d1) 
                    zhibiao[dataset[46]].append(d2)

                    zhibiao[dataset[47]].append((d0*c0*m0+d1*c1*m1+d2*c2*m2)/vis_r_n_2[0])  
                    vis_r_f_2= [(d0*c0*m0+d1*c1*m1+d2*c2*m2),vis_r_n_2[0]] 
                    vis_r_zi=(vis_r_f_jilu[model][ceshi][0]+vis_r_f_2[0])/(vis_r_f_jilu[model][ceshi][1]+vis_r_f_2[1])                
                    zhibiao[dataset[48]].append(vis_r_zi)
            elif ren=='object':
                if ceshi=='success_rate':
                    a0=data[model][ren]['{}_severity_0'.format(dataset[49])][ceshi]
                    a1=data[model][ren]['{}_severity_0'.format(dataset[50])][ceshi]
                    a2=data[model][ren]['{}_severity_0'.format(dataset[51])][ceshi]

                    n0=data[model][ren]['{}_severity_0'.format(dataset[49])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[50])][a_num]
                    n2=data[model][ren]['{}_severity_0'.format(dataset[51])][a_num]
                   

                    zhibiao[dataset[49]].append(a0)
                    zhibiao[dataset[50]].append(a1) 
                    zhibiao[dataset[51]].append(a2)
                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0
                  
                    if (n0+n1+n2)!=0:
                        zhibiao[dataset[52]].append((a0*n0+a1*n1+a2*n2)/(n0+n1+n2))
                        vis_h_n_1=[(a0*n0+a1*n1+a2*n2),(n0+n1+n2)]
                        zhibiao[dataset[53]].append((a0*n0+a1*n1+a2*n2)/(n0+n1+n2))
                    else:
                        zhibiao[dataset[52]].append('none')
                        vis_h_n_1=[(a0*n0+a1*n1+a2*n2),(n0+n1+n2)]
                        zhibiao[dataset[53]].append('none')
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[49])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[50])][ceshi]
                    b2=data[model][ren]['{}_severity_0'.format(dataset[51])][ceshi]
                   

                    zhibiao[dataset[49]].append(b0)
                    zhibiao[dataset[50]].append(b1)
                    zhibiao[dataset[51]].append(b2)
                    
                    if vis_h_n_1[0]!=0:
                        zhibiao[dataset[52]].append((b0*a0*n0+b1*a1*n1+b2*a2*n2)/vis_h_n_1[0])    
                        vis_h_f_1= [(b0*a0*n0+b1*a1*n1+b2*a2*n2),vis_h_n_1[0]] 
                        vis_h_zi=(vis_h_f_1[0])/(vis_h_f_1[1])    
                        zhibiao[dataset[53]].append(vis_h_zi)
                    else:
                        zhibiao[dataset[52]].append("-")    
                        vis_h_f_1= [(b0*a0*n0+b1*a1*n1+b2*a2*n2),vis_h_n_1[0]] 
                        vis_h_zi='-'   
                        zhibiao[dataset[53]].append(vis_h_zi)
            elif ren=='imagenetvc':
                if ceshi=='success_rate':
                    a0=data[model][ren]['{}_severity_0'.format(dataset[54])][ceshi]
                    a1=data[model][ren]['{}_severity_0'.format(dataset[55])][ceshi]
                    a2=data[model][ren]['{}_severity_0'.format(dataset[56])][ceshi]
                    a3=data[model][ren]['{}_severity_0'.format(dataset[57])][ceshi]
                    a4=data[model][ren]['{}_severity_0'.format(dataset[58])][ceshi]

                    n0=data[model][ren]['{}_severity_0'.format(dataset[54])][a_num]
                    n1=data[model][ren]['{}_severity_0'.format(dataset[55])][a_num]
                    n2=data[model][ren]['{}_severity_0'.format(dataset[56])][a_num]
                    n3=data[model][ren]['{}_severity_0'.format(dataset[57])][a_num]
                    n4=data[model][ren]['{}_severity_0'.format(dataset[58])][a_num]
                    

                    zhibiao[dataset[54]].append(a0)
                    zhibiao[dataset[55]].append(a1) 
                    zhibiao[dataset[56]].append(a2)
                    zhibiao[dataset[57]].append(a3) 
                    zhibiao[dataset[58]].append(a4)

                    if a0==0 or a0==-100:
                        a0=0
                    if a1==0 or a1==-100:
                        a1=0
                    if a2==0 or a2==-100:
                        a2=0
                    if a3==0 or a3==-100:
                        a3=0     
                    if a4==0 or a4==-100:
                        a4=0      

                    zhibiao[dataset[59]].append((a0*n0+a1*n1+a2*n2+a3*n3+a4*n4)/(n0+n1+n2+n3+n4))
                    vis_c_n_1=[(a0*n0+a1*n1+a2*n2+a3*n3+a4*n4),(n0+n1+n2+n3+n4)]
                    zhibiao[dataset[60]].append((a0*n0+a1*n1+a2*n2+a3*n3+a4*n4)/(n0+n1+n2+n3+n4))
                else:
                    b0=data[model][ren]['{}_severity_0'.format(dataset[54])][ceshi]
                    b1=data[model][ren]['{}_severity_0'.format(dataset[55])][ceshi]
                    b2=data[model][ren]['{}_severity_0'.format(dataset[56])][ceshi]
                    b3=data[model][ren]['{}_severity_0'.format(dataset[57])][ceshi]
                    b4=data[model][ren]['{}_severity_0'.format(dataset[58])][ceshi]                   

                    zhibiao[dataset[54]].append(b0)
                    zhibiao[dataset[55]].append(b1) 
                    zhibiao[dataset[56]].append(b2)
                    zhibiao[dataset[57]].append(b3) 
                    zhibiao[dataset[58]].append(b4)

                    zhibiao[dataset[59]].append((b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4)/(vis_c_n_1[0]))    
                    vis_c_f_1= [(b0*a0*n0+b1*a1*n1+b2*a2*n2+b3*a3*n3+b4*a4*n4),(vis_c_n_1[0])] 
                    vis_c_zi=(vis_c_f_1[0])/(vis_c_f_1[1])    
                    zhibiao[dataset[60]].append(vis_c_zi)

analysis_df = pd.DataFrame({
    'Model_name': Model_name,
    'ceshi_name': ceshi_name,    
})
for k in range(len(dataset)):
    analysis_df[dataset[k]]=zhibiao[dataset[k]]
                
analysis_df.to_excel('imageattack240305.xlsx', index=False)








                   
                     








    

    
    
    


