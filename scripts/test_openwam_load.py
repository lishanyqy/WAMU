from pathlib import Path
import argparse, tempfile, torch
from omegaconf import OmegaConf

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt-dir',type=Path,required=True); p.add_argument('--wan-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    with tempfile.TemporaryDirectory(prefix='openwam_load_') as td:
        td=Path(td); cfg=OmegaConf.load(a.ckpt_dir/'config.yaml'); cfg.model.video_backbone.model_path=str(a.wan_dir); cfg.model.video_backbone.encoder.model_path=str(a.wan_dir); OmegaConf.save(cfg,td/'config.yaml'); (td/'checkpoint_step_1.safetensors').symlink_to(a.ckpt_dir/'checkpoint_step_9890.safetensors'); (td/'normalization_stats.npy').symlink_to(a.ckpt_dir/'normalization_stats.npy')
        print('loading',flush=True); cfg,arch=load_from_checkpoint_dir(str(td),device=a.device,ckpt_name='checkpoint_step_1.safetensors')
    print('loaded', 'device='+str(next(arch.parameters()).device), 'dtype='+str(next(arch.parameters()).dtype), 'cuda_mem='+str(torch.cuda.memory_allocated()), flush=True)
    del arch; torch.cuda.empty_cache()
if __name__=='__main__': main()
