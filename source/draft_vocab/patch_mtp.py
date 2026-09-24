#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Adapted from MiaAI Lab's patch_mtp_draft_vocab.py (Copyright 2026 MiaAI Lab).
"""Add an opt-in reduced head to the drafter; leave target/logit paths intact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HELPER = '''

def _attach_reduced_draft_vocab(model: nn.Module) -> None:
    path = os.environ.get("VLLM_MTP_DRAFT_VOCAB", "").strip()
    if not path:
        return
    head = model.lm_head
    weight = head.weight
    if getattr(head, "tp_size", 1) != 1:
        raise ValueError("Reduced draft vocabulary requires TP=1")
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("Reduced draft vocabulary expects an unchanged BF16 head")
    if getattr(model.logits_processor, "scale", 1.0) <= 0:
        raise ValueError("Reduced draft vocabulary requires positive logit scale")
    with open(path) as handle:
        ids = sorted({int(line) for line in handle if line.strip()})
    vocab_size = int(getattr(head, "org_vocab_size", weight.shape[0]))
    if not ids or ids[0] < 0 or ids[-1] >= vocab_size or len(ids) >= vocab_size:
        raise ValueError("Invalid reduced draft vocabulary token IDs")
    index = torch.tensor(ids, dtype=torch.long, device=weight.device)
    model.register_buffer("_draft_vocab_weight", weight.detach().index_select(0,index).contiguous(), persistent=False)
    model.register_buffer("_draft_vocab_target_ids", index, persistent=False)
    _draft_vocab_logger.info("Reduced draft vocabulary ACTIVE: %d/%d rows; target/full draft logits unchanged; head dtype %s",len(ids),vocab_size,weight.dtype)

'''
METHOD = '''
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "_draft_vocab_weight", None)
        if weight is None:
            return self.compute_logits(hidden_states).argmax(dim=-1)
        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
        return self._draft_vocab_target_ids[logits.argmax(dim=-1)]

'''

def patch(source):
    assert '_attach_reduced_draft_vocab' not in source
    imports='from vllm.compilation.decorators import support_torch_compile\n'
    anchor='def _remap_ignored_layers(\n'
    compute='    def compute_logits(\n        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n    ) -> torch.Tensor | None:\n        return self.logits_processor(self.lm_head, hidden_states)\n'
    load='        return loader.load_weights(remap_weight_names())\n'
    edits=[(imports,'import os\nfrom vllm.logger import init_logger as _init_draft_vocab_logger\n'+imports),
           (anchor,'_draft_vocab_logger = _init_draft_vocab_logger(__name__)\n'+HELPER+anchor),
           (compute,compute+METHOD),
           (load,'        loaded = loader.load_weights(remap_weight_names())\n        _attach_reduced_draft_vocab(self)\n        return loaded\n')]
    for old,new in edits:
        assert source.count(old)==1,('Nonunique or missing patch anchor',old[:80])
        source=source.replace(old,new)
    ast.parse(source)
    return source

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--manifest',required=True);a=p.parse_args()
    source=Path(a.source);before=source.read_bytes();after=patch(before.decode()).encode()
    source.write_bytes(after)
    Path(a.manifest).write_text(json.dumps({'source':str(source),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'target_changes':False},indent=2))
