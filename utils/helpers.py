import math
import torch
import torch.nn.functional as F
import logging
from pathlib import Path

_logger = logging.getLogger('train')


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    # Copied from `timm` by Ross Wightman:
    # github.com/rwightman/pytorch-image-models
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def pe_check(model, state_dict, pe_key='classifier.positional_emb'):
    if pe_key is not None and pe_key in state_dict.keys() and pe_key in model.state_dict().keys():
        if model.state_dict()[pe_key].shape != state_dict[pe_key].shape:
            state_dict[pe_key] = resize_pos_embed(state_dict[pe_key],
                                                  model.state_dict()[pe_key],
                                                  num_tokens=model.classifier.num_tokens)
    return state_dict


def fc_check(model, state_dict, fc_key='classifier.fc'):
    for key in [f'{fc_key}.weight', f'{fc_key}.bias']:
        if key is not None and key in state_dict.keys() and key in model.state_dict().keys():
            if model.state_dict()[key].shape != state_dict[key].shape:
                _logger.warning(f'Removing {key}, number of classes has changed.')
                state_dict[key] = model.state_dict()[key]
    return state_dict

def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.

    Raises
    ------
    UserWarning
        If commit SHA do not match, or if the directory doesn't have
        the recorded commit SHA.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))
    """
    if len(checkpoint_commit_sha) == 0:
        warnings.warn(
            "Commit SHA was not recorded while saving checkpoints."
        )
    else:
        # verify commit sha, raise warning if it doesn't match
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )
    """
    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]
