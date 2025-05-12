import copy
import math
import random
import os
import csv
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gomoku_env import GomokuEnv

class PolicyValueNet(nn.Module):
    """
    Prosty CNN policy-value network.
    Wejście: 1 x board_size x board_size
    Wyjście:
      - policy: board_size*board_size logits
      - value: pojedynczy skalar w [-1, 1]
    """
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, board_size * board_size)
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1).float()
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.view(-1)

class MCTSNode:
    """
    Węzeł w drzewie MCTS z priors z sieci.
    """
    def __init__(self, state, parent=None, prior=0.0, action=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.priors = None

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, priors):
        self.priors = priors
        for action, p in enumerate(priors.tolist()):
            if self.state.board.flat[action] == 0:
                self.children[action] = MCTSNode(self.state, parent=self, prior=p, action=action)

    def select(self, c_puct):
        total_visits = sum(child.visits for child in self.children.values())
        best_score, best_action, best_child = -float('inf'), None, None
        for act, child in self.children.items():
            u = c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, act, child
        return best_action, best_child

    def backup(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

class MCTSAgent:
    def __init__(self, env: GomokuEnv, net: PolicyValueNet, simulations=800, c_puct=1.0, device='gpu'):
        self.env = env
        self.net = net.to(device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    def choose_action(self, env):
        root = MCTSNode(env)
        obs = torch.tensor(env.board, device=self.device).unsqueeze(0)
        policy_logits, _ = self.net(obs)
        policy = torch.softmax(policy_logits, dim=-1)[0]
        root.expand(policy)
        for _ in range(self.simulations):
            node = root
            sim_env = copy.deepcopy(env)
            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                sim_env.step(action)
            board_copy = sim_env.board.copy()
            done = not (board_copy == 0).any() or node.state._winner is not None
            if not done:
                obs_leaf = torch.tensor(sim_env.board, device=self.device).unsqueeze(0)
                policy_logits, value = self.net(obs_leaf)
                policy = torch.softmax(policy_logits, dim=-1)[0]
                node.expand(policy)
                leaf_value = value.item()
            else:
                leaf_value = 1 if node.state._winner == env.current_player else -1 if node.state._winner else 0
            node.backup(leaf_value)
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

class Trainer:
    """
    Trener sieci i MCTS w self-play z okresową ewaluacją vs losowy gracz.
    Zapisuje statystyki ewaluacji do pliku CSV, w tym średni loss i avg Q.
    """
    def __init__(self, board_size=15, simulations=400, lr=1e-3, device='cpu', stats_file='stats.csv', memory_size=10000):
        self.env = GomokuEnv(board_size=board_size)
        self.net = PolicyValueNet(board_size)
        self.agent = MCTSAgent(self.env, self.net, simulations=simulations, device=device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.device = device
        self.stats_file = stats_file
        self.memory = deque(maxlen=memory_size)  # Bufor pamięci
        # Przygotuj plik statystyk
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'self_play_games', 'eval_games', 'memory_size', 'wins', 'losses', 'draws', 'avg_loss', 'avg_q'])

    def augment_data(self, states, pis):
        augmented_states = []
        augmented_pis = []
        for state, pi in zip(states, pis):
            state = state.reshape(self.env.board_size, self.env.board_size)
            pi = pi.reshape(self.env.board_size, self.env.board_size)
            for k in range(4):  # Obrót o 0, 90, 180, 270 stopni
                rotated_state = np.rot90(state, k)
                rotated_pi = np.rot90(pi, k)
                augmented_states.append(rotated_state.flatten())
                augmented_pis.append(rotated_pi.flatten())
                # Odbicie lustrzane
                flipped_state = np.fliplr(rotated_state)
                flipped_pi = np.fliplr(rotated_pi)
                augmented_states.append(flipped_state.flatten())
                augmented_pis.append(flipped_pi.flatten())
        return augmented_states, augmented_pis

    def self_play(self, games=100):
        print(f"Rozpoczynam self-play dla {games} gier...")
        memory = []
        for game_idx in range(games):
            print(f"Gra self-play {game_idx + 1}/{games}...")
            states, pis, rewards = [], [], []
            obs, _ = self.env.reset()
            done = False
            while not done:
                root = MCTSNode(self.env)
                obs_tensor = torch.tensor(self.env.board, device=self.device).unsqueeze(0)
                logits, _ = self.net(obs_tensor)
                policy = torch.softmax(logits, dim=-1)[0]
                root.expand(policy)
                for __ in range(self.agent.simulations):
                    node = root
                    sim_env = copy.deepcopy(self.env)
                    while not node.is_leaf():
                        action, node = node.select(self.agent.c_puct)
                        sim_env.step(action)
                    obs_leaf = torch.tensor(sim_env.board, device=self.device).unsqueeze(0)
                    logits_leaf, value_leaf = self.net(obs_leaf)
                    priors = torch.softmax(logits_leaf, dim=-1)[0]
                    node.expand(priors)
                    node.backup(value_leaf.item())
                pi_target = np.zeros(self.env.action_space.n, dtype=np.float32)
                for act, child in root.children.items():
                    pi_target[act] = child.visits
                pi_target /= pi_target.sum()
                action = max(root.children.items(), key=lambda item: item[1].visits)[0]
                states.append(self.env.board.copy())
                pis.append(pi_target)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                rewards.append(reward)
            game_value = rewards[-1]
            states, pis = self.augment_data(states, pis)  # Augmentacja danych
            for i in range(len(states)):
                memory.append((states[i], pis[i], game_value))
            print(f"Gra self-play {game_idx + 1}/{games} zakończona.")
        print("Self-play zakończone.")
        self.memory.extend(memory)  # Dodaj do bufora pamięci

    def update_network(self, batch_size=64, epochs=5):
        print(f"Rozpoczynam aktualizację sieci z {len(self.memory)} przykładami w pamięci...")
        self.net.train()
        loss_sum = 0.0
        loss_count = 0
        q_sum = 0.0
        q_count = 0
        for epoch in range(epochs):
            print(f"Epoka {epoch + 1}/{epochs}...")
            for _ in range(0, len(self.memory), batch_size):
                batch = random.sample(self.memory, min(len(self.memory), batch_size))
                states = torch.tensor(np.array([b[0] for b in batch]), device=self.device).float()
                pis = torch.tensor([b[1] for b in batch], device=self.device)
                vs = torch.tensor([b[2] for b in batch], device=self.device).float()

                logits, values = self.net(states)
                q_sum += values.detach().mean().item()
                q_count += 1
                loss_p = - (pis * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss_v = nn.functional.mse_loss(values, vs)
                loss = loss_p + loss_v
                loss_sum += loss.item()
                loss_count += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        avg_loss = loss_sum / loss_count if loss_count else 0
        avg_q = q_sum / q_count if q_count else 0
        print(f"Aktualizacja sieci zakończona. Średni loss: {avg_loss:.4f}, Średni Q: {avg_q:.4f}")
        return avg_loss, avg_q

    def evaluate_random(self, games=20):
        print(f"Rozpoczynam ewaluację przeciwko losowemu graczowi dla {games} gier...")
        results = {'win': 0, 'loss': 0, 'draw': 0}
        for game_idx in range(games):
            print(f"Gra ewaluacyjna {game_idx + 1}/{games}...")
            obs, _ = self.env.reset()
            done = False
            turn = 0
            while not done:
                if turn % 2 == 0:
                    action = self.agent.choose_action(self.env)
                else:
                    legal = [a for a in range(self.env.action_space.n) if self.env.board.flat[a] == 0]
                    action = random.choice(legal)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                turn += 1
            if reward == 1:
                results['win'] += 1
            elif reward == -1:
                results['loss'] += 1
            else:
                results['draw'] += 1
            print(f"Gra ewaluacyjna {game_idx + 1}/{games} zakończona.")
        print(f"Ewaluacja zakończona. Wyniki: {results}")
        return results

    def evaluate_against_previous(self, games=20, previous_model_path='best_model.pth'):
        print(f"Rozpoczynam ewaluację przeciwko wcześniejszej wersji modelu dla {games} gier...")
        # Wczytaj wcześniejszy model
        previous_net = PolicyValueNet(self.env.board_size)
        previous_net.load_state_dict(torch.load(previous_model_path))
        previous_net.to(self.device)
        previous_agent = MCTSAgent(self.env, previous_net, simulations=self.agent.simulations, device=self.device)
        
        results = {'win': 0, 'loss': 0, 'draw': 0}
        for game_idx in range(games):
            print(f"Gra ewaluacyjna {game_idx + 1}/{games}...")
            obs, _ = self.env.reset()
            done = False
            turn = 0
            while not done:
                if turn % 2 == 0:
                    action = self.agent.choose_action(self.env)  # Aktualny model
                else:
                    action = previous_agent.choose_action(self.env)  # Wcześniejszy model
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                turn += 1
            if reward == 1:
                results['win'] += 1
            elif reward == -1:
                results['loss'] += 1
            else:
                results['draw'] += 1
            print(f"Gra ewaluacyjna {game_idx + 1}/{games} zakończona.")
        print(f"Ewaluacja zakończona. Wyniki: {results}")
        return results

    def train(self, iterations=10, self_play_games=20, eval_games=20, model_path='best_model.pth'):
        best_win_rate = 0.0
        for it in range(1, iterations+1):
            print(f"Rozpoczynam iterację {it}...")
            self.self_play(games=self_play_games)
            print(f"Self-play zakończone. Liczba przykładów w pamięci: {len(self.memory)}")
            avg_loss, avg_q = self.update_network()
            print(f"Aktualizacja sieci zakończona. avg_loss={avg_loss:.4f}, avg_q={avg_q:.4f}")
            eval_res = self.evaluate_against_previous(games=eval_games, previous_model_path=model_path)
            print(f"Ewaluacja zakończona. Wyniki: {eval_res}")
            win_rate = eval_res['win'] / eval_games
            print(f"Współczynnik wygranych: {win_rate:.2%}")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(self.net.state_dict(), model_path)
                print(f"Nowy model zapisany jako {model_path} (lepszy wynik).")
            else:
                print("Nowy model nie jest lepszy. Poprzedni model pozostaje bez zmian.")
            print(f"Zapisuję statystyki do pliku {self.stats_file}...")
            with open(self.stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([it, self_play_games, eval_games, len(self.memory),
                                  eval_res['win'], eval_res['loss'], eval_res['draw'],
                                  avg_loss, avg_q])
            print(f"Iteracja {it} zakończona.\n")

if __name__ == '__main__':
    trainer = Trainer(board_size=15, simulations=200, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(iterations=5, self_play_games=10, eval_games=20)
