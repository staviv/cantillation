
NEWDATA = True
ADDTOKENS = True
NIKUD = False # False to remove the nikud
JUST_TEAMIM = False # if True, will remove all the text that is not teamim!
BASE_CHAR = "@"
NUSACHIM =  ["ashkenazi", "maroko", "yerushalmi", "bavly"] #["ashkenazi", "maroko", "yerushalmi", "bavly"]

#load a little data for testing
FASTTEST = False
BATCH_SIZE = 8

SR = 16000
RANDOM = True 
AUGMENT = False

LR = 1e-5
WARMUP_STEPS = 50
EVAL_STEPS = 1000 
SAVE_STEPS = 4000
MAX_STEPS = 8000
DROPOUT = False # False or a number between 0 and 1
WEIGHT_DECAY = False # False or a number

EVALUATE_FIRST_STEP = True # if True, will evaluate the model after the first step

#base model 
BASE_MODEL_VERSIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] # for v3 we need to change the log-mel spectrum
BASE_MODEL_VERSION = BASE_MODEL_VERSIONS[5] # num of model. 0=tiny 1=base... 6=large-v3
BASE_MODEL_NAME = "openai/whisper-" + BASE_MODEL_VERSION
# BASE_MODEL_NAME = "cantillation/Teamim-AllNusah-whisper-medium_Random-True_Mid"
# BASE_MODEL_NAME = "ivrit-ai/whisper-v2-d3-e3" # best hebrew model (large-v2 fine-tuned model)



# current date and time
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")


RUN_NAME = BASE_MODEL_VERSION + "_Random-" + str(RANDOM) + (("_DropOut-" + str(DROPOUT)) if DROPOUT else "") \
            + (("_WeightDecay-" + str(WEIGHT_DECAY)) if WEIGHT_DECAY else "")  + "_Augmented"*AUGMENT \
            + ("" if NEWDATA else "_OldData") + "_date-" + dt_string  # + "_Warmup_steps-" + str(WARMUP_STEPS) + "_Eval_steps-" + str(EVAL_STEPS) + "_Save_steps-" + str(SAVE_STEPS) + "_Max_steps-" + str(MAX_STEPS) # + "_EvalFirstStep-" + str(EVALUATE_FIRST_STEP)  "_LR-" + str(LR) + 




#the new model - after training 
MODEL_NAME = f"./Teamim-{RUN_NAME}" # because the run name doesn't work, I added it to the model name

