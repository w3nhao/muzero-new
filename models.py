from abc import ABC, abstractmethod
from x_transformers import TransformerWrapper, Decoder, Encoder, ContinuousTransformerWrapper

import torch
import torch.nn as nn

class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "SimpleNet":
            return SimpleNetwork(
                observation_shape=config.observation_shape,
                stacked_observations=config.stacked_observations,
                action_space_size=len(config.action_space),
                encoding_size=config.encoding_size,
                max_theory_length=config.max_theory_length,
                dynamics_encode_length=config.dynamics_encode_length,
                trm_encoder_layers=config.trm_encoder_layers,
                trm_attn_heads=config.trm_attn_heads,
                fusion_mlp_layers=config.fusion_mlp_layers,
                fc_reward_layers=config.fc_reward_layers,
                fc_value_layers=config.fc_value_layers,
                fc_policy_layers=config.fc_policy_layers,
                fc_dynamics_layers=config.fc_dynamics_layers,
                support_size=config.support_size,
            )
        elif config.network == "ComplexNet":
            # return ComplexNetwork(config)
            pass
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def masked_mean(tensor, mask, dim):
        mask = mask.unsqueeze(-1).expand_as(tensor)
        # Apply mask to tensor
        tensor_masked = tensor * mask
        sum_tensor = tensor_masked.sum(dim=dim)
        mask_sum = mask.sum(dim=dim)
        masked_mean = sum_tensor / mask_sum.clamp(min=1)
        return masked_mean


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class ConvEncoderWithGN(nn.Module):
    def __init__(self, in_channels, out_channels, out_len):
        super(ConvEncoderWithGN, self).__init__()
        # Initial convolution to reduce dimensionality
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 2, 
                               kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels // 2)
        self.relu = nn.ReLU()
        
        # Second convolution for further processing
        self.conv2 = nn.Conv1d(in_channels=out_channels // 2, out_channels=out_channels, 
                               kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        
        # Adaptive pooling to resize to the desired output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_len)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.adaptive_pool(x)  # Resize feature maps to the desired output length
        return x



class SimpleRepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        encoding_size,
        max_theory_length,
        dynamics_encode_length,
        trm_encoder_layers,
        trm_attn_heads,
        fusion_mlp_layers,
    ):
        super().__init__()  
        self.max_theory_length = max_theory_length
        
        self.theory_encoder = TransformerWrapper(
            num_tokens=action_space_size,
            max_seq_len=max_theory_length,
            attn_layers = Encoder(
            dim = encoding_size,
            depth = trm_encoder_layers,
            heads = trm_attn_heads,
            use_simple_rmsnorm = True,
            attn_flash = True,
            rotary_pos_emb = True,
            )
        )
        
        self.dynamics_encoder = ContinuousTransformerWrapper(
            dim_in = 32,
            dim_out = encoding_size,
            max_seq_len = dynamics_encode_length,
            attn_layers = Encoder(
                dim = encoding_size,
                depth = trm_encoder_layers,
                heads = trm_attn_heads,
                rotary_pos_emb = True,
                attn_flash = True,
            )
        )

        self.dynamics_embedder = ConvEncoderWithGN(
            in_channels=observation_shape[0],
            out_channels=32,
            out_len=dynamics_encode_length,
        )
        
        self.mlp = mlp(
            input_size=encoding_size * 2,
            layer_sizes=fusion_mlp_layers,
            output_size=encoding_size,
        )
            
        
    def forward(self, theory, mask, dynamics):
        theory_encoded = self.theory_encoder(theory, mask=mask)
        dynamics_encoded = self.dynamics_encoder(self.dynamics_embedder(dynamics))
        # mean pooling over length dimension
        dynamics_encoded = dynamics_encoded.mean(dim=1)
        # theory should be mean pooled over length dimension but mask should be applied
        theory_encoded = masked_mean(theory_encoded, mask, dim=1)
        encoded_state = torch.cat((theory_encoded, dynamics_encoded), dim=1)
        encoded_state = self.mlp(encoded_state)
        return encoded_state
        

class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

class SimpleNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        max_theory_length,
        dynamics_encode_length,
        trm_encoder_layers,
        trm_attn_heads,
        fusion_mlp_layers,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        
        self.representation_network = torch.nn.DataParallel(
            SimpleRepresentationNetwork(
                observation_shape=observation_shape,
                action_space_size=action_space_size,
                encoding_size=encoding_size,
                max_theory_length=max_theory_length,
                dynamics_encode_length=dynamics_encode_length,
                trm_encoder_layers=trm_encoder_layers,
                trm_attn_heads=trm_attn_heads,
                fusion_mlp_layers=fusion_mlp_layers,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state




def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
