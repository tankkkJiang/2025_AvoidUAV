from stable_baselines3.common.callbacks import BaseCallback
import wandb, os

class WandbCallback(BaseCallback):
    """
    自定义 W&B 回调：
    1. 每步把 SB3 logger 中的标量同步到 wandb
    2. 每 save_freq 步保存一个模型 checkpoint
    3. 统计 episode 级别指标并上传
    """
    def __init__(self, save_freq:int, save_path:str, verbose:int=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        # 记录成功率
        self.ep_cnt, self.ep_success = 0, 0

    # 每个环境 step 后都会调用
    def _on_step(self) -> bool:
        # -------- 1. 同步 SB3 logger 内置标量 --------
        # SB3 把最近一 batch 的 scalar 存在 self.model.logger.name_to_value
        scalar_names = [
            'train/policy_loss', 'train/value_loss',
            'train/entropy_loss', 'train/approx_kl', 'train/clip_fraction'
        ]
        metrics = {k:self.model.logger.name_to_value.get(k, 0.0) for k in scalar_names}

        # -------- 2. 检查 episode 是否结束 --------
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode", False):
                self.ep_cnt += 1
                ep_info = info["episode"]
                # 例如 episode_reward、length
                metrics.update({f"episode/{k}":v for k,v in ep_info.items()})
                if info.get("arrival", False):
                    self.ep_success += 1
                metrics["episode/success_rate"] = self.ep_success / self.ep_cnt

        # 上传到 wandb
        wandb.log(metrics, step=self.num_timesteps)

        # -------- 3. 定期存 checkpoint --------
        if self.num_timesteps % self.save_freq == 0:
            ckpt = os.path.join(self.save_path, f"ckpt_{self.num_timesteps}.zip")
            self.model.save(ckpt)
            if self.verbose:
                print(f"[wandb] checkpoint 保存至 {ckpt}")

        return True