"""all model trainer"""

from typing import Union, List, Callable

from aiak_training_llm.models import get_model_family


MODEL_FAMILY_TRAINER_FACTORY = {}

# decorator factory to register model trainers
def register_model_trainer(model_family: Union[str, List[str]], training_phase: str, training_func: Callable = None):
    """
    register model training function

    Args:
        model_family: need to be consistent with the models.factory definition, otherwise it
                      cannot be retrieved correctly. (Case-insensitive)

        training_phase: need to be consistent with the --training-phase definition in train.arguments
        trainig_func: training function. 
    """
    # add to the factory
    def _add_trainer(families, phase, func):
        #convert to list if single string
        if not isinstance(families, list): 
            families = [families]

        #loop through all families
        for _family in families:
            _family = _family.lower() # make it case-insensitive
            #create entry for this family if not exist
            if _family not in MODEL_FAMILY_TRAINER_FACTORY:
                MODEL_FAMILY_TRAINER_FACTORY[_family] = {}
            #check for duplicate registration (family+phase)
            if phase in MODEL_FAMILY_TRAINER_FACTORY[_family]:
                raise ValueError(f"Cannot register duplicate trainer ({_family} family, {phase} phase)")
            #register the trainer function
            MODEL_FAMILY_TRAINER_FACTORY[_family][phase] = func
        
    def _register_function(fn):
        _add_trainer(model_family, training_phase, fn)
        return fn #fn is the training function being decorated

    if training_func is not None:
        return _add_trainer(model_family, training_phase, training_func)
    else:
        return _register_function


def build_model_trainer(args): # all args including model name and training phase
    """create model trainer"""

    # get model family name
    #args.model_name="llava-ov-1.5-4b" in our case
    model_family = get_model_family(args.model_name)    #map model name to model family
    print(f"Model family: {model_family}")
    #returns family name like model_family="llava_ov_1_5" in our case
    # get model family trainer

    #check if this model family has a registered trainer
    # MODEL_FAMILY_TRAINER_FACTORY is a nested dictionary ={family:{phase:trainer_func}}
    if model_family not in MODEL_FAMILY_TRAINER_FACTORY:
        raise ValueError(f"Not found trainer for {args.model_name} (family: {model_family})")
    
    # check if this training phase has a registered trainer for this model family
    # args.training_phase="sft" in our case
    print(f"Training phase: {args.training_phase}")
    if args.training_phase not in MODEL_FAMILY_TRAINER_FACTORY[model_family]:
        raise ValueError(f"AIAK not support {args.training_phase} phase for {args.model_name} (family: {model_family})")
    
    #retrieve the trainer function for this model family and training phase from the factory
    # MODEL_FAMILY_TRAINER_FACTORY[llava_ov_1_5][sft] -> returns the trainer function
    trainer = MODEL_FAMILY_TRAINER_FACTORY[model_family][args.training_phase]
    # This retrieves the default_pretrain_trainer function defined in sft_llavaov_1_5_vl.py
    return trainer(args)  # Calls default_pretrain_trainer(args)
    # returns a trainer object for the specified model and training phase
    #trainer(args) calls the trainer function with args.
    #trainer function creates a megatron trainer object and returns it.
