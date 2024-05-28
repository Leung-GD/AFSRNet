import torch
from MSC_net_cov import *

model=torch.load("/media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/MSC-net_org_3ch_64/paper_ASOSR/baeline_new/0.15ASR_50_1000000_512liberty_aug/0.15ASR_45_1000000_512.pth")
torch.save(model.state_dict(),"MSCDesc.pth")