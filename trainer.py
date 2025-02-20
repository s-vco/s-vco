from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
import torch.nn.functional as F


class SVCOTrainerForLlava(DPOTrainer):
    def __init__(self, beta_img_win_vs_no_img_preference, beta_no_img_vs_img_lose_preference, beta_img_lose_vs_no_img_preference, beta_no_img_vs_img_win_preference, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_img_win_vs_no_img_preference, self.beta_no_img_vs_img_lose_preference, \
        self.beta_img_lose_vs_no_img_preference, self.beta_no_img_vs_img_win_preference, \
            = (
            beta_img_win_vs_no_img_preference,
            beta_no_img_vs_img_lose_preference,
            beta_img_lose_vs_no_img_preference,
            beta_no_img_vs_img_win_preference,
        )
        self.PAD_TOKEN_ID, self.IGNORE_TOKEN_ID, self.label_pad_token_id = (
            self.tokenizer.pad_token_id, -100, -100
        )

    def concatenated_inputs(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        (
            chosen_pixel_values, chosen_input_ids, chosen_labels, chosen_attention_mask,
            rejected_pixel_values, rejected_input_ids, rejected_labels, rejected_attention_mask,
            no_img_chosen_input_ids, no_img_chosen_labels, no_img_chosen_attention_mask,
            no_img_rejected_input_ids, no_img_rejected_labels, no_img_rejected_attention_mask
        ) = batch

        # has_img
        has_img_concatenated_batch = {
            "pixel_values": torch.cat([
                chosen_pixel_values, rejected_pixel_values, chosen_pixel_values, rejected_pixel_values
            ], dim=0),
            "input_ids": torch.cat([
                chosen_input_ids, chosen_input_ids, rejected_input_ids, rejected_input_ids
            ], dim=0),
            "labels": torch.cat([
                chosen_labels, chosen_labels, rejected_labels, rejected_labels
            ], dim=0),
            "attention_mask": torch.cat([
                chosen_attention_mask, chosen_attention_mask, rejected_attention_mask, rejected_attention_mask
            ], dim=0),
        }

        # no_img
        no_img_concatenated_batch = {
            "input_ids": torch.cat([no_img_chosen_input_ids, no_img_rejected_input_ids], dim=0),
            "labels": torch.cat([no_img_chosen_labels, no_img_rejected_labels], dim=0),
            "attention_mask": torch.cat([no_img_chosen_attention_mask, no_img_rejected_attention_mask], dim=0),
        }
        return has_img_concatenated_batch, no_img_concatenated_batch

    def concatenated_forward(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, ...]:
        bs_chosen = batch[1].shape[0]    # chosen_input_ids
        bs_rejected = batch[5].shape[0]  # rejected_input_ids

        has_img_batch, no_img_batch = self.concatenated_inputs(batch)
        
        # convert dtype
        model_dtype = next(model.parameters()).dtype
        has_img_batch = {
            k: v.to(model_dtype) if "float" in str(v.dtype) else v
            for k, v in has_img_batch.items()
        }
        
        # forward pass (has_img)
        has_img_out = model(**has_img_batch, return_dict=True)
        has_img_logits = has_img_out["logits"]
        has_img_labels = has_img_out["labels_for_loss_calculation"]
        has_img_logps = self._get_batch_logps(has_img_logits, has_img_labels, average_log_prob=False)

        # (bs*4)
        logps_img_chosen_res_chosen = has_img_logps[:bs_chosen]
        logps_img_rejected_res_chosen = has_img_logps[bs_chosen : bs_chosen + bs_chosen]
        logps_img_chosen_res_rejected = has_img_logps[bs_chosen + bs_chosen : bs_chosen + bs_chosen + bs_rejected]
        logps_img_rejected_res_rejected = has_img_logps[bs_chosen + bs_chosen + bs_rejected :]

        logits_img_chosen_res_chosen = has_img_logits[:bs_chosen]
        logits_img_rejected_res_chosen = has_img_logits[bs_chosen : bs_chosen + bs_chosen]
        logits_img_chosen_res_rejected = has_img_logits[bs_chosen + bs_chosen : bs_chosen + bs_chosen + bs_rejected]
        logits_img_rejected_res_rejected = has_img_logits[bs_chosen + bs_chosen + bs_rejected :]

        # no img
        no_img_batch = {
            k: v.to(model_dtype) if "float" in str(v.dtype) else v
            for k, v in no_img_batch.items()
        }
        no_img_out = model(**no_img_batch, return_dict=True)
        no_img_logits = no_img_out["logits"]
        no_img_labels = no_img_out["labels_for_loss_calculation"]
        no_img_logps = self._get_batch_logps(no_img_logits, no_img_labels, average_log_prob=False)

        # (bs*2)
        logps_no_img_res_chosen = no_img_logps[:bs_chosen]
        logps_no_img_res_rejected = no_img_logps[bs_chosen:]

        logits_no_img_res_chosen = no_img_logits[:bs_chosen]
        logits_no_img_res_rejected = no_img_logits[bs_chosen:]

        return (
            logps_img_chosen_res_chosen,
            logps_img_rejected_res_chosen,
            logps_img_chosen_res_rejected,
            logps_img_rejected_res_rejected,
            logps_no_img_res_chosen,
            logps_no_img_res_rejected,
            logits_img_chosen_res_chosen,
            logits_img_rejected_res_chosen,
            logits_img_chosen_res_rejected,
            logits_img_rejected_res_rejected,
            logits_no_img_res_chosen,
            logits_no_img_res_rejected
        )

    def dpo_loss(
        self,
        policy_logps_img_chosen_res_chosen: torch.FloatTensor,
        policy_logps_img_rejected_res_chosen: torch.FloatTensor,
        policy_logps_img_chosen_res_rejected: torch.FloatTensor,
        policy_logps_img_rejected_res_rejected: torch.FloatTensor,
        policy_logps_no_img_res_chosen: torch.FloatTensor,
        policy_logps_no_img_res_rejected: torch.FloatTensor,
        reference_logps_img_chosen_res_chosen: torch.FloatTensor,
        reference_logps_img_rejected_res_chosen: torch.FloatTensor,
        reference_logps_img_chosen_res_rejected: torch.FloatTensor,
        reference_logps_img_rejected_res_rejected: torch.FloatTensor,
        reference_logps_no_img_res_chosen: torch.FloatTensor,
        reference_logps_no_img_res_rejected: torch.FloatTensor,
        reference_free: bool = False,
        train_eval: str = "train",
    ):
        # (res_win|img_win) > (res_win|no_img)
        pi_img_win_vs_no_img = policy_logps_img_chosen_res_chosen - policy_logps_no_img_res_chosen
        ref_img_win_vs_no_img = (
            0 if reference_free else reference_logps_img_chosen_res_chosen - reference_logps_no_img_res_chosen
        )
        logits_img_win_vs_no_img = pi_img_win_vs_no_img - ref_img_win_vs_no_img
        dist_img_win_vs_no_img = (policy_logps_img_chosen_res_chosen - policy_logps_no_img_res_chosen).abs()

        # (res_win|no_img) > (res_win|img_lose)
        pi_no_img_vs_img_lose = policy_logps_no_img_res_chosen - policy_logps_img_rejected_res_chosen
        ref_no_img_vs_img_lose = (
            0 if reference_free else reference_logps_no_img_res_chosen - reference_logps_img_rejected_res_chosen
        )
        logits_no_img_vs_img_lose = pi_no_img_vs_img_lose - ref_no_img_vs_img_lose
        dist_no_img_vs_img_lose = (policy_logps_no_img_res_chosen - policy_logps_img_rejected_res_chosen).abs()

        # (res_lose|img_lose) > (res_lose|no_img)
        pi_img_lose_vs_no_img = policy_logps_img_rejected_res_rejected - policy_logps_no_img_res_rejected
        ref_img_lose_vs_no_img = (
            0 if reference_free else reference_logps_img_rejected_res_rejected - reference_logps_no_img_res_rejected
        )
        logits_img_lose_vs_no_img = pi_img_lose_vs_no_img - ref_img_lose_vs_no_img
        dist_img_lose_vs_no_img = (policy_logps_img_rejected_res_rejected - policy_logps_no_img_res_rejected).abs()

        # (res_lose|no_img) > (res_lose|img_win)
        pi_no_img_vs_img_win = policy_logps_no_img_res_rejected - policy_logps_img_chosen_res_rejected
        ref_no_img_vs_img_win = (
            0 if reference_free else reference_logps_no_img_res_rejected - reference_logps_img_chosen_res_rejected
        )
        logits_no_img_vs_img_win = pi_no_img_vs_img_win - ref_no_img_vs_img_win
        dist_no_img_vs_img_win = (policy_logps_no_img_res_rejected - policy_logps_img_chosen_res_rejected).abs()

        if train_eval == "train":
            img_win_vs_no_img_loss = (
                -F.logsigmoid(self.beta_img_win_vs_no_img_preference * logits_img_win_vs_no_img)
                if self.beta_img_win_vs_no_img_preference > 0
                else torch.tensor(0.0, requires_grad=False)
            )
            no_img_vs_img_lose_loss = (
                -F.logsigmoid(self.beta_no_img_vs_img_lose_preference * logits_no_img_vs_img_lose)
                if self.beta_no_img_vs_img_lose_preference > 0
                else torch.tensor(0.0, requires_grad=False)
            )
            img_lose_vs_no_img_loss = (
                -F.logsigmoid(self.beta_img_lose_vs_no_img_preference * logits_img_lose_vs_no_img)
                if self.beta_img_lose_vs_no_img_preference > 0
                else torch.tensor(0.0, requires_grad=False)
            )
            no_img_vs_img_win_loss = (
                -F.logsigmoid(self.beta_no_img_vs_img_win_preference * logits_no_img_vs_img_win)
                if self.beta_no_img_vs_img_win_preference > 0
                else torch.tensor(0.0, requires_grad=False)
            )

            losses = (
                img_win_vs_no_img_loss
                + no_img_vs_img_lose_loss
                + img_lose_vs_no_img_loss
                + no_img_vs_img_win_loss
            )
        
        elif train_eval == "eval":
            # beta defaults
            img_win_beta = self.beta_img_win_vs_no_img_preference if self.beta_img_win_vs_no_img_preference > 0 else 0.1
            no_img_lose_beta = self.beta_no_img_vs_img_lose_preference if self.beta_no_img_vs_img_lose_preference > 0 else 0.1
            img_lose_beta = self.beta_img_lose_vs_no_img_preference if self.beta_img_lose_vs_no_img_preference > 0 else 0.1
            no_img_win_beta = self.beta_no_img_vs_img_win_preference if self.beta_no_img_vs_img_win_preference > 0 else 0.1

            img_win_vs_no_img_loss = -F.logsigmoid(img_win_beta * logits_img_win_vs_no_img)
            no_img_vs_img_lose_loss = -F.logsigmoid(no_img_lose_beta * logits_no_img_vs_img_lose)
            img_lose_vs_no_img_loss = -F.logsigmoid(img_lose_beta * logits_img_lose_vs_no_img)
            no_img_vs_img_win_loss = -F.logsigmoid(no_img_win_beta * logits_no_img_vs_img_win)

            losses = (
                img_win_vs_no_img_loss
                + no_img_vs_img_lose_loss
                + img_lose_vs_no_img_loss
                + no_img_vs_img_win_loss
            )

        # Rewards
        rewards_img_chosen_res_chosen = (policy_logps_img_chosen_res_chosen - reference_logps_img_chosen_res_chosen).detach()
        rewards_img_rejected_res_chosen = (policy_logps_img_rejected_res_chosen - reference_logps_img_rejected_res_chosen).detach()
        rewards_no_img_res_chosen = (policy_logps_no_img_res_chosen - reference_logps_no_img_res_chosen).detach()
        rewards_img_rejected_res_rejected = (policy_logps_img_rejected_res_rejected - reference_logps_img_rejected_res_rejected).detach()
        rewards_img_chosen_res_rejected = (policy_logps_img_chosen_res_rejected - reference_logps_img_chosen_res_rejected).detach()
        rewards_no_img_res_rejected = (policy_logps_no_img_res_rejected - reference_logps_no_img_res_rejected).detach()

        return {
            # losses
            "img_win_vs_no_img_loss": img_win_vs_no_img_loss,
            "no_img_vs_img_lose_loss": no_img_vs_img_lose_loss,
            "img_lose_vs_no_img_loss": img_lose_vs_no_img_loss,
            "no_img_vs_img_win_loss": no_img_vs_img_win_loss,
            "losses": losses,
            # rewards
            "rewards_img_chosen_res_chosen": rewards_img_chosen_res_chosen,
            "rewards_img_rejected_res_chosen": rewards_img_rejected_res_chosen,
            "rewards_no_img_res_chosen": rewards_no_img_res_chosen,
            "rewards_img_rejected_res_rejected": rewards_img_rejected_res_rejected,
            "rewards_img_chosen_res_rejected": rewards_img_chosen_res_rejected,
            "rewards_no_img_res_rejected": rewards_no_img_res_rejected,
            # distances
            "pi_logratios_img_win_vs_no_img_distance": dist_img_win_vs_no_img,
            "pi_logratios_no_img_vs_img_lose_distance": dist_no_img_vs_img_lose,
            "pi_logratios_img_lose_vs_no_img_distance": dist_img_lose_vs_no_img,
            "pi_logratios_no_img_vs_img_win_distance": dist_no_img_vs_img_win,
        }

    def get_batch_metrics(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        metrics = {}
        # policy forward
        (
            policy_logps_img_chosen_res_chosen,
            policy_logps_img_rejected_res_chosen,
            policy_logps_img_chosen_res_rejected,
            policy_logps_img_rejected_res_rejected,
            policy_logps_no_img_res_chosen,
            policy_logps_no_img_res_rejected,
            policy_logits_img_chosen_res_chosen,
            policy_logits_img_rejected_res_chosen,
            policy_logits_img_chosen_res_rejected,
            policy_logits_img_rejected_res_rejected,
            policy_logits_no_img_res_chosen,
            policy_logits_no_img_res_rejected,
        ) = self.concatenated_forward(model, batch)

        # reference forward
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_logps_img_chosen_res_chosen,
                        reference_logps_img_rejected_res_chosen,
                        reference_logps_img_chosen_res_rejected,
                        reference_logps_img_rejected_res_rejected,
                        reference_logps_no_img_res_chosen,
                        reference_logps_no_img_res_rejected,
                        reference_logits_img_chosen_res_chosen,
                        reference_logits_img_rejected_res_chosen,
                        reference_logits_img_chosen_res_rejected,
                        reference_logits_img_rejected_res_rejected,
                        reference_logits_no_img_res_chosen,
                        reference_logits_no_img_res_rejected,
                    ) = self.concatenated_forward(model, batch)
            else:
                (
                    reference_logps_img_chosen_res_chosen,
                    reference_logps_img_rejected_res_chosen,
                    reference_logps_img_chosen_res_rejected,
                    reference_logps_img_rejected_res_rejected,
                    reference_logps_no_img_res_chosen,
                    reference_logps_no_img_res_rejected,
                    reference_logits_img_chosen_res_chosen,
                    reference_logits_img_rejected_res_chosen,
                    reference_logits_img_chosen_res_rejected,
                    reference_logits_img_rejected_res_rejected,
                    reference_logits_no_img_res_chosen,
                    reference_logits_no_img_res_rejected,
                ) = self.concatenated_forward(self.ref_model, batch)

        # S-VCO Loss
        svco_loss_info = self.dpo_loss(
            policy_logps_img_chosen_res_chosen,
            policy_logps_img_rejected_res_chosen,
            policy_logps_img_chosen_res_rejected,
            policy_logps_img_rejected_res_rejected,
            policy_logps_no_img_res_chosen,
            policy_logps_no_img_res_rejected,
            reference_logps_img_chosen_res_chosen,
            reference_logps_img_rejected_res_chosen,
            reference_logps_img_chosen_res_rejected,
            reference_logps_img_rejected_res_rejected,
            reference_logps_no_img_res_chosen,
            reference_logps_no_img_res_rejected,
            reference_free=False,
            train_eval=train_eval,
        )

        # loss & reward info
        img_win_vs_no_img_loss = svco_loss_info["img_win_vs_no_img_loss"]
        no_img_vs_img_lose_loss = svco_loss_info["no_img_vs_img_lose_loss"]
        img_lose_vs_no_img_loss = svco_loss_info["img_lose_vs_no_img_loss"]
        no_img_vs_img_win_loss = svco_loss_info["no_img_vs_img_win_loss"]
        losses = svco_loss_info["losses"]

        rewards_img_chosen_res_chosen = svco_loss_info["rewards_img_chosen_res_chosen"]
        rewards_img_rejected_res_chosen = svco_loss_info["rewards_img_rejected_res_chosen"]
        rewards_no_img_res_chosen = svco_loss_info["rewards_no_img_res_chosen"]
        rewards_img_rejected_res_rejected = svco_loss_info["rewards_img_rejected_res_rejected"]
        rewards_img_chosen_res_rejected = svco_loss_info["rewards_img_chosen_res_rejected"]
        rewards_no_img_res_rejected = svco_loss_info["rewards_no_img_res_rejected"]

        dist_img_win_vs_no_img = svco_loss_info["pi_logratios_img_win_vs_no_img_distance"]
        dist_no_img_vs_img_lose = svco_loss_info["pi_logratios_no_img_vs_img_lose_distance"]
        dist_img_lose_vs_no_img = svco_loss_info["pi_logratios_img_lose_vs_no_img_distance"]
        dist_no_img_vs_img_win = svco_loss_info["pi_logratios_no_img_vs_img_win_distance"]

        # reward accuracies
        img_win_vs_no_img_reward_accuracies = (rewards_img_chosen_res_chosen > rewards_no_img_res_chosen).float()
        no_img_vs_img_lose_reward_accuracies = (rewards_no_img_res_chosen > rewards_img_rejected_res_chosen).float()
        img_lose_vs_no_img_reward_accuracies = (rewards_img_rejected_res_rejected > rewards_no_img_res_rejected).float()
        no_img_vs_img_win_reward_accuracies = (rewards_no_img_res_rejected > rewards_img_chosen_res_rejected).float()

        # reward margins
        img_win_vs_no_img_reward_margins = (rewards_img_chosen_res_chosen - rewards_no_img_res_chosen).float()
        no_img_vs_img_lose_reward_margins = (rewards_no_img_res_chosen - rewards_img_rejected_res_chosen).float()
        img_lose_vs_no_img_reward_margins = (rewards_img_rejected_res_rejected - rewards_no_img_res_rejected).float()
        no_img_vs_img_win_reward_margins = (rewards_no_img_res_rejected - rewards_img_chosen_res_rejected).float()

        # log metrics
        prefix = f"{train_eval}_"
        # losses
        metrics[f"{prefix}losses/img_win_vs_no_img_loss"] = img_win_vs_no_img_loss.mean().cpu()
        metrics[f"{prefix}losses/no_img_vs_img_lose_loss"] = no_img_vs_img_lose_loss.mean().cpu()
        metrics[f"{prefix}losses/img_lose_vs_no_img_loss"] = img_lose_vs_no_img_loss.mean().cpu()
        metrics[f"{prefix}losses/no_img_vs_img_win_loss"] = no_img_vs_img_win_loss.mean().cpu()
        # rewards
        metrics[f"{prefix}rewards/rewards_img_chosen_res_chosen"] = rewards_img_chosen_res_chosen.mean().cpu()
        metrics[f"{prefix}rewards/rewards_img_rejected_res_chosen"] = rewards_img_rejected_res_chosen.mean().cpu()
        metrics[f"{prefix}rewards/rewards_no_img_res_chosen"] = rewards_no_img_res_chosen.mean().cpu()
        metrics[f"{prefix}rewards/rewards_img_rejected_res_rejected"] = rewards_img_rejected_res_rejected.mean().cpu()
        metrics[f"{prefix}rewards/rewards_img_chosen_res_rejected"] = rewards_img_chosen_res_rejected.mean().cpu()
        metrics[f"{prefix}rewards/rewards_no_img_res_rejected"] = rewards_no_img_res_rejected.mean().cpu()
        # reward accuracies
        metrics[f"{prefix}reward_acc/img_win_vs_no_img_reward_accuracies"] = img_win_vs_no_img_reward_accuracies.mean().cpu()
        metrics[f"{prefix}reward_acc/no_img_vs_img_lose_reward_accuracies"] = no_img_vs_img_lose_reward_accuracies.mean().cpu()
        metrics[f"{prefix}reward_acc/img_lose_vs_no_img_reward_accuracies"] = img_lose_vs_no_img_reward_accuracies.mean().cpu()
        metrics[f"{prefix}reward_acc/no_img_vs_img_win_reward_accuracies"] = no_img_vs_img_win_reward_accuracies.mean().cpu()
        # reward margins
        metrics[f"{prefix}reward_margin/img_win_vs_no_img_reward_margins"] = img_win_vs_no_img_reward_margins.mean().cpu()
        metrics[f"{prefix}reward_margin/no_img_vs_img_lose_reward_margins"] = no_img_vs_img_lose_reward_margins.mean().cpu()
        metrics[f"{prefix}reward_margin/img_lose_vs_no_img_reward_margins"] = img_lose_vs_no_img_reward_margins.mean().cpu()
        metrics[f"{prefix}reward_margin/no_img_vs_img_win_reward_margins"] = no_img_vs_img_win_reward_margins.mean().cpu()
        # logps
        metrics[f"{prefix}logps/policy_logps_img_chosen_res_chosen"] = policy_logps_img_chosen_res_chosen.mean().cpu()
        metrics[f"{prefix}logps/policy_logps_img_rejected_res_chosen"] = policy_logps_img_rejected_res_chosen.mean().cpu()
        metrics[f"{prefix}logps/policy_logps_img_chosen_res_rejected"] = policy_logps_img_chosen_res_rejected.mean().cpu()
        metrics[f"{prefix}logps/policy_logps_img_rejected_res_rejected"] = policy_logps_img_rejected_res_rejected.mean().cpu()
        metrics[f"{prefix}logps/policy_logps_no_img_res_chosen"] = policy_logps_no_img_res_chosen.mean().cpu()
        metrics[f"{prefix}logps/policy_logps_no_img_res_rejected"] = policy_logps_no_img_res_rejected.mean().cpu()
        # distances
        metrics[f"{prefix}pi_logratios/pi_logratios_img_win_vs_no_img_distance"] = dist_img_win_vs_no_img.mean().cpu()
        metrics[f"{prefix}pi_logratios/pi_logratios_no_img_vs_img_lose_distance"] = dist_no_img_vs_img_lose.mean().cpu()
        metrics[f"{prefix}pi_logratios/pi_logratios_img_lose_vs_no_img_distance"] = dist_img_lose_vs_no_img.mean().cpu()
        metrics[f"{prefix}pi_logratios/pi_logratios_no_img_vs_img_win_distance"] = dist_no_img_vs_img_win.mean().cpu()

        return losses.mean(), metrics

