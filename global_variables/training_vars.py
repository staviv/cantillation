COMBINE_DATA = False # if True, will combine the old data with the new data
NEWDATA = True # "other" # True, False, "other" (for other data)
ADDTOKENS = True
NIKUD = False # False to remove the nikud
JUST_TEAMIM = False # if True, will remove all the text! that is not teamim!
BASE_CHAR = "@"
NUSACHIM =  ["ashkenazi", "maroko", "yerushalmi", "bavly"] # ["ashkenazi", "maroko", "yerushalmi", "bavly"]


FASTTEST = False # load small data for testing the code.
BATCH_SIZE = 32


SR = 16000
RANDOM = False 
AUGMENT = True # if True, will augment the data 

LR = 1e-5
WARMUP_STEPS = 1000
EVAL_STEPS = 1000
SAVE_STEPS = 1000
MAX_STEPS = 20000
DROPOUT = False # False or a number between 0 and 1 TODO: fix it in the code 
WEIGHT_DECAY = 0.005 # False or a number between 0 and 1 (recommended between 0.01 and 0.1. [based on tests I did])
INIT_OUTPUT_LAYER = False # if True, will initialize the output layer with the base model weights.

EVALUATE_FIRST_STEP = False # if True, will evaluate the model after the first step

#base model 
BASE_MODEL_VERSIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] # for v3 we need to change the log-mel spectrum
BASE_MODEL_VERSION = BASE_MODEL_VERSIONS[3] # num of model. 0=tiny 1=base... 6=large-v3
BASE_MODEL_NAME = "openai/whisper-" + BASE_MODEL_VERSION
# BASE_MODEL_NAME = "cantillation/Teamim-large-v2_WeightDecay-0.05_Augmented_Combined-Data_date-14-07-2024_18-24"
# BASE_MODEL_NAME = "ivrit-ai/whisper-v2-d3-e3"
# BASE_MODEL_NAME = "ivrit-ai/whisper-v2-pd1-e1" # best hebrew model (large-v2 fine-tuned model)

# current date and time
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y")

if COMBINE_DATA:
    datatype = "Combined"
else:
    datatype = "New" if NEWDATA else "Old"

RUN_NAME = BASE_MODEL_VERSION + ("_Random" if RANDOM else "") + (("_DropOut-" + str(DROPOUT)) if DROPOUT else "") \
            + (("_WeightDecay-" + str(WEIGHT_DECAY)) if WEIGHT_DECAY else "")  + "_Augmented"*AUGMENT \
             + "_WithNikud"*NIKUD + "_" + datatype + "-Data" + "_date-" + dt_string
            # + "_Warmup_steps-" + str(WARMUP_STEPS) + "_Eval_steps-" + str(EVAL_STEPS) + "_Save_steps-" + str(SAVE_STEPS) + "_Max_steps-" + str(MAX_STEPS) # + "_EvalFirstStep-" + str(EVALUATE_FIRST_STEP) + "_LR-" + str(LR) + 




#the new model - after training 
MODEL_NAME = f"./Teamim-{RUN_NAME}" # because the run name doesn't work, I added it to the model name

