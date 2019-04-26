from set_up_translation import *

t = get_translation_objects('.en', '.en')
print(t['train_data'].examples[0].__dict__)