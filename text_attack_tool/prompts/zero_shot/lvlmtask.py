# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains the prompt sets for the zeroshot role-oriented prompts.
"""
"""
You are a prompt assistant. Please give me 9 prompts with the same meaning as the input prompt. Your output should be a list containing nine prompts. "{}" Means that the location of the content can be added in Prompt. Do not add "{}" to the Prompt without "{}".
For example:
Input prompt:"As a mathematics instructor, calculate the answer to the following problem related to {}: ".
You output:["Pretend you are a math teacher and solve this mathematical question concerning {}: ",
        "Imagine being a math mentor and determine the solution to the following problem with {}: ",
        "In the position of a mathematics coach, figure out the answer to this math challenge involving {}: ",
        "Act as a math expert and compute the result of the following mathematical task about {}: ",
        "As a mathematics consultant, resolve the following problem associated with {}: ",
        "Taking on the role of a math specialist, work out the solution for this mathematical question featuring {}: ",
        "Embodying a mathematics advisor, find the answer to the following math problem with {}: ",
        "As a math guru, evaluate the following mathematical task regarding {}: ",
        "In the shoes of a mathematics master, obtain the result for this math question that includes {}: "].
The new input prompt is:"What is described in the image?", what should you output?
"""


#CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet
LVLM_PROMPT_SET = {
    'ScienceQAIMG':["{}Choose the best answer from the following choices:{}",
        "{}Select the most suitable answer from the following options:{}",
"{}From the choices provided below, choose the best answer:{}",
"{}Consider the following options and select the correct answer:{}",
"{}Evaluate the given choices and determine the best answer:{}",
"{}Among the following options, identify the correct answer:{}",
"{}Review the following choices and indicate the best answer:{}",
"{}Examine the provided options and choose the most appropriate answer:{}",
"{}Take a look at the following choices and select the best answer:{}",
"{}Analyze the given options and determine the most suitable answer:{}"],

'WHOOPSWeird':['What is unusual about this image? Please provide a concise one-sentence explanation.',"Provide a brief one-sentence explanation about what makes this image unusual.",
"Summarize in a single sentence what is unique about this image.",
"Offer a concise one-sentence explanation for what sets this image apart.",
"Describe in one sentence what is distinct or peculiar about this image.",
"Give a brief one-sentence explanation highlighting the unusual aspect of this image.",
"State in a concise one-sentence explanation what makes this image different.",
"Provide a one-sentence summary of what is noteworthy about this image.",
"Summarize the unusual characteristic of this image in a single sentence.",
"Offer a concise one-sentence explanation for what is out of the ordinary in this image."],

'IconQA':["{}Choose the best answer from the following choices:{}",
        "{}Select the most suitable answer from the following options:{}",
"{}From the choices provided below, choose the best answer:{}",
"{}Consider the following options and select the correct answer:{}",
"{}Evaluate the given choices and determine the best answer:{}",
"{}Among the following options, identify the correct answer:{}",
"{}Review the following choices and indicate the best answer:{}",
"{}Examine the provided options and choose the most appropriate answer:{}",
"{}Take a look at the following choices and select the best answer:{}",
"{}Analyze the given options and determine the most suitable answer:{}"],

'AOKVQAClose':[
"{}Choose the best answer from the following choices:{}",
        "{}Select the most suitable answer from the following options:{}",
"{}From the choices provided below, choose the best answer:{}",
"{}Consider the following options and select the correct answer:{}",
"{}Evaluate the given choices and determine the best answer:{}",
"{}Among the following options, identify the correct answer:{}",
"{}Review the following choices and indicate the best answer:{}",
"{}Examine the provided options and choose the most appropriate answer:{}",
"{}Take a look at the following choices and select the best answer:{}",
"{}Analyze the given options and determine the most suitable answer:{}"],

    'object':['Is there {} in the image?',"Does the image contain {}?",
"Can you identify {} in the image?",
"Is there {} visible in the image?",
"Do you see {} in the image?",
"Is {} present in the image?",
"Can you spot {} in the image?",
"Is there {} depicted in the image?",
"Do you notice {} in the image?",
"Is {} observable in the image?"],
    'Flowers102': ["What breed is the flower in the image?",
        "Identify the breed of the flower shown in the image.",
        "Name the type of flower that is in the image.",
        "Determine the species of the flower in the picture.",
        "Recognize the breed of the flower depicted in the image.",
        "Specify the type of flower that is shown in the picture.",
        "Figure out the name of the flower in the image.",
        "State the breed of the flower that is visible in the photograph.",
        "Provide the name of the flower species in the picture.",
        "Describe the breed of the flower that is shown in the image."],
# I want you to act as a prompt generator for squad v2 dataset.  
# Here is an example : "Please provide the most accurate answer based on the context. If the answer cannot be found in the context, respond with 'unanswerable'.". " 
# Please generate 10 similar prompts. the prompt is used for MMLU (Measuring Massive Multitask Language Understanding) dataset.  
# For the prompts, please first add a quote " at the beginning and the end of each sentence, and then and a comma at the end.
    'OxfordIIITPet':["What breed is the pet in the image?","Identify the breed of the animal in the picture.",
        "Name the type of animal in the image.",
        "Tell me the breed of the pet shown in the picture.",
        "Determine the breed of the animal in the photo.",
        "Specify the breed of the pet in the image.",
        "Recognize the type of animal in the picture and name its breed.",
        "What is the breed of the pet featured in the image?",
        "Describe the breed of the animal in the photo.",
        "Provide the name of the breed of the pet in the image."],

    'CIFAR10': ["The photo of the","The image of the", "The picture of the", "The photograph of the", "The snapshot of the", "The shot of the", "The portrait of the", "The still of the", "The frame of the", "The visual of the"],

    'CIFAR100': ["The photo of the","The image of the", "The picture of the", "The photograph of the", "The snapshot of the", "The shot of the", "The portrait of the", "The still of the", "The frame of the", "The visual of the"],

    'ImageNet': ["The photo of the","The image of the", "The picture of the", "The photograph of the", "The snapshot of the", "The shot of the", "The portrait of the", "The still of the", "The frame of the", "The visual of the"],

    'OCR': ["What is written in the image?",
        "Can you describe what you see in the image?",
        "Provide a verbal depiction of what is portrayed in the image.",
        "What visual elements are present in the image?",
        "Tell me about the contents of the image.",
        "What can you observe in the image?",
        "Please describe the image in words.",
        "What is depicted in the image?",
        "Describe the visual information in the image.",
        "What is the image showing?"],

    # 56.34, 66.20, 61.97, 59.15, 59.15, 56.34, 64.79, 57.75, 64.79, 54.93
    'caption': ["what is described in the image?",
        "Describe what you see in the image.",
        "Provide a written description of the image.",
        "Explain the contents of the image.",
        "Write a description of what is depicted in the image.",
        "Interpret the image and provide a description of its contents.",
        "Give a verbal depiction of the image.",
        "Provide a detailed description of the image.",
        "Describe the visual elements of the image.",
        "Provide a written account of what is shown in the image."],

    # 84.48, 84.12, 84.48, 84.48, 84.12, 84.84, 84.84, 83.03, 85.56, 82.31
    'rte': [
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
        "Acting as an entailment detection instrument, determine if the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",
        "As a tool for determining entailment relationships, review the two statements and categorize their connection as either 'entailment' or 'not_entailment':",
        "While performing entailment analysis, classify the relationship between the provided sentences as 'entailment' or 'not_entailment':",
        "In the capacity of an entailment assessment system, indicate if the link between the following sentences is 'entailment' or 'not_entailment':",
        "Working as an entailment classifier, identify whether the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment':",
        "As an instrument for entailment evaluation, consider the two sentences and determine if their relationship is 'entailment' or 'not_entailment'. Respond with 'entailment' or 'not_entailment':",
        "In the role of a semantic relationship analyzer, examine the connection between the given sentences and decide if they exhibit entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",       
    ],

    'mnli': [
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment', 'neutral', or 'contradiction':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction':",
        "Acting as an entailment detection instrument, determine if the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':",
        "As a tool for determining entailment relationships, review the two statements and categorize their connection as either 'entailment', 'neutral', or 'contradiction':",
        "While performing entailment analysis, classify the relationship between the provided sentences as 'entailment', 'neutral', or 'contradiction':",
        "In the capacity of an entailment assessment system, indicate if the link between the following sentences is 'entailment', 'neutral', or 'contradiction':",
        "Working as an entailment classifier, identify whether the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':",
        "As an instrument for entailment evaluation, consider the two sentences and determine if their relationship is 'entailment', 'neutral', or 'contradiction':",
        "In the role of a semantic relationship analyzer, examine the connection between the given sentences and decide if they exhibit entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':",
    ],

    'cola': [
        "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect:",
        "As a grammar identification system, examine the provided sentence and respond with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
        "Functioning as a grammar evaluation tool, analyze the given sentence and decide if it is grammatically correct, responding with 'acceptable' or 'unacceptable':",
        "Acting as a grammar detection instrument, determine if the provided sentence is grammatically sound, answering with 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "As a tool for determining grammatical correctness, review the sentence and categorize its grammar as either 'acceptable' or 'unacceptable':",
        "While performing grammar analysis, classify the grammar of the following sentence as 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "In the capacity of a grammar assessment system, indicate if the structure of the provided sentence is grammatically correct, responding with 'acceptable' or 'unacceptable':",
        "Working as a grammar classifier, identify whether the given sentence has correct grammar, and respond with 'acceptable' for correct sentences or 'unacceptable' for incorrect ones:",
        "As an instrument for grammar evaluation, consider the sentence and determine if its grammar is correct, responding with 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "In the role of a syntax analyzer, examine the grammar of the provided sentence and decide if it is correct, answering with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
    ],

    'qqp': [
        "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'. ",
        "As a question equivalence detection system, examine the provided questions and respond with 'equivalent' if they are the same in meaning, or 'not_equivalent' if they are different. ",
        "Functioning as a question similarity evaluation tool, analyze the given questions and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
        "Acting as a question equivalence instrument, determine if the provided questions are equivalent in meaning, answering with 'equivalent' for similar questions or 'not_equivalent' for dissimilar ones. ",
        "As a tool for determining question equivalence, review the questions and categorize their similarity as either 'equivalent' or 'not_equivalent'. ",
        "While performing question comparison analysis, classify the similarity of the following questions as 'equivalent' for equivalent questions or 'not_equivalent' for different questions. ",
        "In the capacity of a question assessment system, indicate if the meaning of the provided questions is the same, responding with 'equivalent' or 'not_equivalent'. ",
        "Working as a question classifier, identify whether the given questions share the same meaning, and respond with 'equivalent' for equivalent questions or 'not_equivalent' for different ones. ",
        "As an instrument for question comparison evaluation, consider the questions and determine if their meaning is the same, responding with 'equivalent' for similar questions or 'not_equivalent' for different questions. ",
        "In the role of a question similarity analyzer, examine the meaning of the provided questions and decide if they are equivalent, answering with 'equivalent' for equivalent questions or 'not_equivalent' for different questions. ",
    ],

    # 86.95, 88.65, 88.85, 87.90, 83.10, 74.45, 88.55, 88.65, 88.85, 83.80
    'qnli': [
        "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'. ",
        "In your role as a semantic evaluator, determine if the provided context justifies the answer to the question and answer with 'entailment' or 'not_entailment'. ",
        "As a textual analyst, examine if the given context logically implies the answer to the question and indicate your decision with 'entailment' or 'not_entailment'. ",
        "As a semantic researcher, evaluate whether the provided context supports the answer to the question and choose 'entailment' or 'not_entailment'. ",
        "In the capacity of a language specialist, decide if the context presented contains enough information to infer the answer to the question and respond with 'entailment' or 'not_entailment'. ",
        "As a textual inference expert, analyze if the answer to the question can be deduced from the provided context and select 'entailment' or 'not_entailment'. ",
        "In your role as a linguistic investigator, determine if the context given entails the answer to the question and provide your conclusion with 'entailment' or 'not_entailment'. ",
        "As a semantic interpreter, assess whether the provided context supports the answer to the given question and answer with 'entailment' or 'not_entailment'. ",
        "In the capacity of a language evaluator, examine if the given context justifies the answer to the question and indicate your assessment with 'entailment' or 'not_entailment'. ",
        "As a linguistic consultant, decide if the answer to the question is logically supported by the provided context and respond with 'entailment' or 'not_entailment'. ",    
    ],


    # 82.60, 77.94, 80.39, 81.13, 80.64, 75.74, 81.62, 81.13, 79.66, 82.60
    'mrpc': [
        "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a language analyst, assess the following sentences and classify their similarity as 'equivalent' or 'not_equivalent'. ",
        "As a sentence similarity evaluator, analyze the provided sentences and indicate if their meanings are 'equivalent' or 'not_equivalent'. ",
        "In the role of a textual comparison specialist, examine the given sentences and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
        "As a linguistic comparator, review the following pair of sentences and determine their semantic equivalence by choosing 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a semantic assessment tool, evaluate the provided sentences and classify their meanings as 'equivalent' or 'not_equivalent'. ",
        "As a language comparison expert, examine the given pair of sentences and decide if their meanings align, answering with 'equivalent' or 'not_equivalent'. ",
        "In the role of a sentence comparison analyst, assess the provided sentences and indicate if they convey the same meaning by selecting 'equivalent' or 'not_equivalent'. ",
        "As a textual similarity evaluator, analyze the following pair of sentences and determine if they are semantically 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a semantic comparison tool, examine the given sentences and decide if their meanings are identical, responding with 'equivalent' or 'not_equivalent'. ", 
    ],
}