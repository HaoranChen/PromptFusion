"""
Author: Haoran Chen
Date: 2024.07.07
"""

import torch
import torch.nn as nn
import clip

class VPT_Prompt(nn.Module):
    def __init__(self, length=20, embed_dim=768, prompt_init='uniform'):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_init = prompt_init

        prompt_pool_shape = (length, embed_dim)
        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)
    
    
    def forward(self, x_embed):
        out = dict()

        batched_prompt = self.prompt.expand(x_embed.shape[0], -1, -1) 
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out

def clip_text_prompt_genarator(classnames, clip_model, prompt):
    length = prompt.shape[1]
    prompt_prefix = " ".join(["X"] * (length * 2 ))

    classnames = [name.replace("_", " ") for name in classnames]
    prompt_with_cls = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompt_with_cls]).to('cuda')

    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts).float()

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + length:, :]

    prompt_full = torch.cat(
        [prefix,  # (n_cls, 1, dim)
         prompt,  # (n_cls, M1, dim)
         suffix,  # (n_cls, *, dim)
         ],
        dim=1)

    return prompt_full, tokenized_prompts
