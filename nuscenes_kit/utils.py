import torch
val_scene = ['scene-0107', 'scene-0018', 'scene-0783', 'scene-1069', 'scene-0557', 'scene-0967', 'scene-0770', 'scene-0962',
             'scene-1071', 'scene-0800', 'scene-0634', 'scene-0098', 'scene-0558', 'scene-0038', 'scene-0905', 'scene-0272',
             'scene-0629', 'scene-0929', 'scene-0102', 'scene-0928', 'scene-0035', 'scene-0268', 'scene-0780', 'scene-0110',
             'scene-0520', 'scene-0907', 'scene-1070', 'scene-0972', 'scene-0561', 'scene-0909', 'scene-0910', 'scene-0922',
             'scene-0104', 'scene-0329', 'scene-0269', 'scene-0917', 'scene-0923', 'scene-0625', 'scene-0930', 'scene-0778',
             'scene-0635', 'scene-0553', 'scene-0559', 'scene-0784', 'scene-0794', 'scene-0330', 'scene-0523', 'scene-0563',
             'scene-0277', 'scene-0799', 'scene-0521', 'scene-0003', 'scene-0636', 'scene-1066', 'scene-1061', 'scene-0906',
             'scene-1068', 'scene-0344', 'scene-0096', 'scene-0094', 'scene-0781', 'scene-0109', 'scene-0637', 'scene-0519',
             'scene-0346', 'scene-0562', 'scene-0630', 'scene-0638', 'scene-0632', 'scene-0524', 'scene-0092', 'scene-0920',
             'scene-0969', 'scene-0912', 'scene-0564', 'scene-0633', 'scene-0915', 'scene-1063', 'scene-0926', 'scene-1065',
             'scene-0221', 'scene-0919', 'scene-0554', 'scene-0552', 'scene-0968', 'scene-0565', 'scene-0913', 'scene-1073',
             'scene-0039', 'scene-0911', 'scene-0271', 'scene-0777', 'scene-0273', 'scene-0095', 'scene-0345', 'scene-0927',
             'scene-0925', 'scene-0775', 'scene-0332', 'scene-0626', 'scene-0015', 'scene-0908', 'scene-0012', 'scene-0103',
             'scene-0099', 'scene-0016', 'scene-0771', 'scene-0916', 'scene-0971', 'scene-0797', 'scene-1064', 'scene-0093',
             'scene-0782', 'scene-1072', 'scene-0924', 'scene-0560', 'scene-0522', 'scene-0904', 'scene-0555', 'scene-0108',
             'scene-0276', 'scene-1067', 'scene-0921', 'scene-0627', 'scene-0966', 'scene-1059', 'scene-0795', 'scene-0796',
             'scene-0914', 'scene-0275', 'scene-0013', 'scene-0097', 'scene-0798', 'scene-0014', 'scene-0101', 'scene-0105',
             'scene-0036', 'scene-0331', 'scene-0106', 'scene-1062', 'scene-0017', 'scene-0270', 'scene-0963', 'scene-0556',
             'scene-0274', 'scene-1060', 'scene-0100', 'scene-0931', 'scene-0802', 'scene-0278']
def process_prompt(processor, qas, images=[], plans=[], locs=[], bos=True):
    conversations = []

    for q, a in qas:
        conversations.append(
            {
                "from": "human",
                "value": q,
            }
        )
        conversations.append(
            {
                "from": "gpt",
                "value": a,
            }
        )

    item = {"image": images, "conversations": conversations, "plan": plans, "loc": locs}

    _prompt = processor.process_item(item, bos=bos)
    prompt = []

    for value in _prompt:
        if isinstance(value, int):
            prompt.append(value)
        else:
            prompt += value["input_ids"]
    prompt = torch.tensor(prompt, dtype=torch.int64, device='cuda').unsqueeze(0)
    return prompt

def encode_mask(processor):
    mask_tokens = ''
    max_frames = 5
    # 3 plan tokens and 2 special tokens
    mask_len = 5 

    for i in range(max_frames):
        for j in range(mask_len):
            mask_tokens += f"<reserved{14696+i}>"
    
    return mask_tokens, process_prompt(processor, [[mask_tokens, None]], bos=False)

def decode_plans(plans):

    ps = plans.split('>')
    ts = list()

    for i, p in enumerate(ps):
        if i % 5 != 0 and i % 5 != 4:
            ts.append(int(p[-5:]))

    decoded = list()

    for i, t in enumerate(ts):
        if i % 3 == 0:
            decoded.append([])
        if i % 3 != 2:
            decoded[-1].append((t-10500)/50.-20)
        else:
            decoded[-1].append((t-15000)/100.-1.7)
    
    return decoded
