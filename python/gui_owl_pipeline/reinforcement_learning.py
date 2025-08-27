"""
Reinforcement Learning module implementing GUI-Owl's Trajectory-aware Relative Policy Optimization (TRPO).
Provides scalable RL framework with trajectory-level rewards and normalized advantage estimation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
from datetime import datetime
from .config import PipelineConfig
from .trajectory_collector import Trajectory
from .enhanced_evaluator import EnhancedTrajectoryEvaluation

logger = logging.getLogger(__name__)

@dataclass
class RLExperience:
    """Single experience for RL training."""
    trajectory_id: str
    states: List[Dict[str, Any]]  # Sequence of states
    actions: List[Dict[str, Any]]  # Sequence of actions taken
    log_probs: List[float]  # Log probabilities of actions
    rewards: float  # Trajectory-level reward
    advantage: float  # Normalized advantage
    value_estimates: List[float]  # Value function estimates
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TRPOConfig:
    """Configuration for TRPO algorithm."""
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 32
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    replay_buffer_size: int = 10000
    positive_sample_ratio: float = 0.3  # Ratio of positive samples to inject
    reward_accuracy_weight: float = 0.7
    reward_format_weight: float = 0.3
    advantage_normalization: bool = True
    use_replay_buffer: bool = True

class PolicyNetwork(nn.Module):
    """
    Neural network for the policy and value functions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value estimate.
        """
        shared_features = self.shared(state)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """Get action probability distribution."""
        policy_logits, _ = self.forward(state)
        return torch.distributions.Categorical(logits=policy_logits)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        Returns log_probs, values, and entropy.
        """
        policy_logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy

class ReplayBuffer:
    """
    Experience replay buffer for storing successful trajectories.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.success_buffer = deque(maxlen=max_size // 2)  # Separate buffer for high-quality trajectories
    
    def add(self, experience: RLExperience, is_success: bool = False):
        """Add experience to buffer."""
        self.buffer.append(experience)
        if is_success:
            self.success_buffer.append(experience)
    
    def sample(self, batch_size: int, positive_ratio: float = 0.3) -> List[RLExperience]:
        """
        Sample batch with specified ratio of positive examples.
        """
        n_positive = int(batch_size * positive_ratio)
        n_regular = batch_size - n_positive
        
        samples = []
        
        # Sample from success buffer if available
        if len(self.success_buffer) > 0 and n_positive > 0:
            positive_samples = np.random.choice(
                list(self.success_buffer),
                size=min(n_positive, len(self.success_buffer)),
                replace=True
            )
            samples.extend(positive_samples)
        
        # Sample from regular buffer
        if len(self.buffer) > 0 and n_regular > 0:
            regular_samples = np.random.choice(
                list(self.buffer),
                size=min(n_regular, len(self.buffer)),
                replace=True
            )
            samples.extend(regular_samples)
        
        return samples
    
    def __len__(self):
        return len(self.buffer)

class TRPOOptimizer:
    """
    Trajectory-aware Relative Policy Optimization (TRPO) implementation.
    """
    
    def __init__(self, config: TRPOConfig, state_dim: int, action_dim: int):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Replay buffer
        if config.use_replay_buffer:
            self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "rewards": []
        }
    
    def compute_trajectory_reward(self,
                                 trajectory: Trajectory,
                                 evaluation: EnhancedTrajectoryEvaluation) -> float:
        """
        Compute trajectory-level reward R(τ).
        R(τ) = accuracy_reward + format_reward
        """
        
        # Accuracy reward based on evaluation
        if evaluation.is_correct:
            accuracy_reward = 1.0
        elif evaluation.consensus_result == "PARTIAL":
            accuracy_reward = 0.5
        else:
            accuracy_reward = 0.0
        
        # Adjust accuracy reward based on confidence
        accuracy_reward *= evaluation.confidence
        
        # Format reward based on trajectory quality
        format_reward = 0.0
        
        # Reward for completing trajectory
        if trajectory.status == "completed":
            format_reward += 0.3
        
        # Reward for good step ratio
        good_steps = sum(1 for e in evaluation.step_evaluations 
                        if e.annotation.value == "GOOD")
        good_ratio = good_steps / len(evaluation.step_evaluations) if evaluation.step_evaluations else 0
        format_reward += good_ratio * 0.4
        
        # Penalty for harmful steps
        harmful_steps = len(evaluation.harmful_steps)
        harmful_penalty = min(harmful_steps * 0.1, 0.3)
        format_reward -= harmful_penalty
        
        # Combine rewards
        total_reward = (
            self.config.reward_accuracy_weight * accuracy_reward +
            self.config.reward_format_weight * format_reward
        )
        
        return max(-1.0, min(1.0, total_reward))  # Clip to [-1, 1]
    
    def compute_normalized_advantages(self, experiences: List[RLExperience]) -> List[float]:
        """
        Compute normalized advantage estimates.
        Â_τ = (R(τ) - R̄) / (σ_R + ε)
        """
        
        rewards = [exp.rewards for exp in experiences]
        
        if not rewards:
            return []
        
        # Compute mean and std
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if len(rewards) > 1 else 1.0
        
        # Normalize advantages
        advantages = []
        for exp in experiences:
            if self.config.advantage_normalization:
                normalized_adv = (exp.rewards - mean_reward) / (std_reward + 1e-8)
            else:
                normalized_adv = exp.rewards - mean_reward
            
            advantages.append(normalized_adv)
        
        return advantages
    
    def prepare_training_batch(self, experiences: List[RLExperience]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch of experiences for training.
        """
        
        # Flatten sequences
        all_states = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        
        # Compute normalized advantages
        advantages = self.compute_normalized_advantages(experiences)
        
        for exp, adv in zip(experiences, advantages):
            # Convert states to feature vectors
            state_features = [self._state_to_features(s) for s in exp.states]
            action_indices = [self._action_to_index(a) for a in exp.actions]
            
            all_states.extend(state_features)
            all_actions.extend(action_indices)
            all_log_probs.extend(exp.log_probs)
            
            # Use the same advantage for all steps in trajectory
            all_advantages.extend([adv] * len(exp.states))
            
            # Compute returns for value function training
            returns = self._compute_returns(exp.rewards, len(exp.states))
            all_returns.extend(returns)
        
        # Convert to tensors
        batch = {
            "states": torch.FloatTensor(all_states).to(self.device),
            "actions": torch.LongTensor(all_actions).to(self.device),
            "old_log_probs": torch.FloatTensor(all_log_probs).to(self.device),
            "advantages": torch.FloatTensor(all_advantages).to(self.device),
            "returns": torch.FloatTensor(all_returns).to(self.device)
        }
        
        return batch
    
    def _state_to_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert state dictionary to feature vector.
        This is a simplified version - in practice would use more sophisticated encoding.
        """
        
        features = []
        
        # Basic state features
        features.append(float(state.get("step_index", 0)) / 100)  # Normalized step index
        features.append(float(len(state.get("windows", []))))  # Window count
        features.append(float(state.get("element_count", 0)) / 1000)  # Normalized element count
        
        # Action history encoding (last 5 actions)
        action_history = state.get("action_history", [])
        for i in range(5):
            if i < len(action_history):
                features.append(float(self._action_to_index(action_history[-(i+1)])))
            else:
                features.append(0.0)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _action_to_index(self, action: Dict[str, Any]) -> int:
        """Convert action dictionary to index."""
        
        action_types = ["click", "type", "scroll", "swipe", "press", "wait", "unknown"]
        action_type = action.get("type", "unknown")
        
        try:
            return action_types.index(action_type)
        except ValueError:
            return action_types.index("unknown")
    
    def _compute_returns(self, trajectory_reward: float, n_steps: int) -> List[float]:
        """
        Compute discounted returns for each step.
        """
        
        # Distribute reward across trajectory with discount
        returns = []
        running_return = trajectory_reward
        
        for i in range(n_steps):
            returns.append(running_return)
            running_return *= self.config.gamma
        
        return returns
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute TRPO loss.
        L_TRPO = -1/N ΣΣΣ min[r_t(θ)Â_τi, clip(r_t(θ), 1-ε, 1+ε)Â_τi]
        """
        
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        # Get current policy predictions
        log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
        
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_coef * value_loss + 
            self.config.entropy_coef * entropy_loss
        )
        
        # Logging statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "mean_ratio": ratio.mean().item(),
            "mean_advantage": advantages.mean().item()
        }
        
        return total_loss, stats
    
    def update(self, experiences: List[RLExperience]) -> Dict[str, float]:
        """
        Perform TRPO update on batch of experiences.
        """
        
        if not experiences:
            return {}
        
        # Add to replay buffer if enabled
        if self.config.use_replay_buffer:
            for exp in experiences:
                is_success = exp.rewards > 0.7
                self.replay_buffer.add(exp, is_success)
        
        # Prepare training batch
        if self.config.use_replay_buffer and len(self.replay_buffer) > self.config.batch_size:
            # Mix current experiences with replay
            replay_experiences = self.replay_buffer.sample(
                self.config.batch_size // 2,
                self.config.positive_sample_ratio
            )
            training_experiences = experiences[:self.config.batch_size // 2] + replay_experiences
        else:
            training_experiences = experiences
        
        batch = self.prepare_training_batch(training_experiences)
        
        # Multiple epochs of updates
        epoch_stats = []
        for epoch in range(self.config.n_epochs):
            # Compute loss
            loss, stats = self.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )
            
            # Update parameters
            self.optimizer.step()
            
            epoch_stats.append(stats)
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Aggregate statistics
        avg_stats = {
            key: np.mean([s[key] for s in epoch_stats])
            for key in epoch_stats[0].keys()
        }
        
        # Update training statistics
        self.training_stats["total_updates"] += 1
        self.training_stats["policy_losses"].append(avg_stats["policy_loss"])
        self.training_stats["value_losses"].append(avg_stats["value_loss"])
        self.training_stats["entropies"].append(avg_stats["entropy"])
        
        logger.info(
            f"TRPO Update {self.training_stats['total_updates']}: "
            f"Policy Loss={avg_stats['policy_loss']:.4f}, "
            f"Value Loss={avg_stats['value_loss']:.4f}, "
            f"Entropy={avg_stats['entropy']:.4f}"
        )
        
        return avg_stats
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "training_stats": self.training_stats
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.old_policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        
        logger.info(f"Loaded checkpoint from {filepath}")

class RLTrainingPipeline:
    """
    Complete RL training pipeline integrating with the data collection system.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.trpo_config = TRPOConfig()
        
        # Initialize TRPO optimizer
        # State dim and action dim are simplified here
        self.optimizer = TRPOOptimizer(
            self.trpo_config,
            state_dim=20,  # Simplified state features
            action_dim=7   # Number of action types
        )
        
        # Experience collection
        self.current_experiences: List[RLExperience] = []
        self.experience_buffer = deque(maxlen=1000)
    
    async def collect_experiences(self,
                                 trajectories: List[Trajectory],
                                 evaluations: List[EnhancedTrajectoryEvaluation]) -> List[RLExperience]:
        """
        Collect RL experiences from trajectories and evaluations.
        """
        
        experiences = []
        
        for traj, eval in zip(trajectories, evaluations):
            # Compute trajectory reward
            reward = self.optimizer.compute_trajectory_reward(traj, eval)
            
            # Extract states and actions
            states = []
            actions = []
            log_probs = []
            
            for i, step in enumerate(traj.steps):
                # Create state representation
                state = {
                    "step_index": i,
                    "windows": step.pre_state.get("windows", []),
                    "element_count": step.pre_state.get("ui_hierarchy", {}).get("element_count", 0),
                    "action_history": [s.action for s in traj.steps[:i]][-5:]  # Last 5 actions
                }
                states.append(state)
                
                # Action
                actions.append(step.action)
                
                # Placeholder log prob (would be computed during rollout)
                log_probs.append(0.0)
            
            # Create experience
            exp = RLExperience(
                trajectory_id=traj.id,
                states=states,
                actions=actions,
                log_probs=log_probs,
                rewards=reward,
                advantage=0.0,  # Will be computed during training
                value_estimates=[],
                metadata={
                    "query": traj.query.natural_instruction,
                    "success": eval.is_correct,
                    "confidence": eval.confidence
                }
            )
            
            experiences.append(exp)
        
        logger.info(f"Collected {len(experiences)} experiences for RL training")
        
        return experiences
    
    async def train_iteration(self,
                            trajectories: List[Trajectory],
                            evaluations: List[EnhancedTrajectoryEvaluation]) -> Dict[str, Any]:
        """
        Perform one iteration of RL training.
        """
        
        # Collect experiences
        new_experiences = await self.collect_experiences(trajectories, evaluations)
        
        # Add to buffer
        self.current_experiences.extend(new_experiences)
        
        # Train if enough experiences
        if len(self.current_experiences) >= self.trpo_config.batch_size:
            # Perform TRPO update
            update_stats = self.optimizer.update(self.current_experiences)
            
            # Clear current experiences
            self.current_experiences = []
            
            return {
                "trained": True,
                "update_stats": update_stats,
                "experiences_used": len(new_experiences)
            }
        else:
            return {
                "trained": False,
                "experiences_collected": len(new_experiences),
                "total_buffered": len(self.current_experiences)
            }
    
    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action probabilities for a given state.
        """
        
        # Convert state to features
        state_features = self.optimizer._state_to_features(state)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.optimizer.device)
        
        # Get policy distribution
        with torch.no_grad():
            dist = self.optimizer.policy.get_action_distribution(state_tensor)
            probs = dist.probs.squeeze(0).cpu().numpy()
        
        # Map to action types
        action_types = ["click", "type", "scroll", "swipe", "press", "wait", "unknown"]
        action_probs = {
            action_types[i]: float(probs[i])
            for i in range(len(action_types))
        }
        
        return action_probs
    
    def select_action(self, state: Dict[str, Any], temperature: float = 1.0) -> str:
        """
        Select an action based on the current policy.
        """
        
        action_probs = self.get_action_probabilities(state)
        
        # Apply temperature for exploration control
        if temperature != 1.0:
            probs = np.array(list(action_probs.values()))
            probs = np.power(probs, 1.0 / temperature)
            probs /= np.sum(probs)
            
            for i, key in enumerate(action_probs.keys()):
                action_probs[key] = probs[i]
        
        # Sample action
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        
        selected_action = np.random.choice(actions, p=probabilities)
        
        return selected_action
    
    def export_training_stats(self, filepath: str):
        """Export training statistics to JSON."""
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_updates": self.optimizer.training_stats["total_updates"],
            "config": self.trpo_config.__dict__,
            "recent_losses": {
                "policy": self.optimizer.training_stats["policy_losses"][-100:],
                "value": self.optimizer.training_stats["value_losses"][-100:],
                "entropy": self.optimizer.training_stats["entropies"][-100:]
            },
            "buffer_stats": {
                "replay_buffer_size": len(self.optimizer.replay_buffer) if self.trpo_config.use_replay_buffer else 0,
                "experience_buffer_size": len(self.experience_buffer)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Exported training statistics to {filepath}")