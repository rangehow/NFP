
from sacrebleu.metrics import BLEU, CHRF, TER

import logging


logger = logging.getLogger(__name__)
log_level = "ERROR"
logger.setLevel(log_level)


ref_file=['/data/ruanjh/best_training_method/wmt22/generaltest2022.en-de.ref.A.de','/data/ruanjh/best_training_method/wmt22/generaltest2022.en-de.ref.B.de']
for ref in ref_file:
    mt=f'/data/ruanjh/best_training_method/wmt22/t5_v3'


    # src_data=open(src).readlines()
    mt_data=open(mt).readlines()
    ref_data=open(ref).readlines()

    # print(len(mt_data),len(ref_data))

    bleu = BLEU()
    print(bleu.corpus_score(mt_data, [ref_data]))

    # chrf = CHRF()
    # print(chrf.corpus_score(mt_data, [ref_data]))
# comet22
# from comet import download_model, load_from_checkpoint
# # data = [
# #     {
# #         "src": "Dem Feuer konnte Einhalt geboten werden",
# #         "mt": "The fire could be stopped",
# #         "ref": "They were able to control the fire."
# #     },
# #     {
# #         "src": "Schulen und Kindergärten wurden eröffnet.",
# #         "mt": "Schools and kindergartens were open",
# #         "ref": "Schools and kindergartens opened"
# #     }
# # ]
#     model = load_from_checkpoint(f'{config.comet_path}')
#     data=[{"src": s,"mt": m,"ref": r}  for s,m,r in  zip(src_data,mt_data,ref_data)]
#     model_output = model.predict(data, batch_size=8, gpus=8)
#     print (f'{file} comet22\n',model_output[-1])
