# Name: Sachin Banjade
# EUID: SB3064

import tkinter as tk
import math
import matplotlib.pyplot as plt

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}
CELL_SIZE = 80
FONT_VALUE = ('Arial', 10)
FONT_LABEL = ('Arial', 16, 'bold')


def parse_gridworld(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    idx = 0
    rows, cols = map(int, lines[idx].split()); idx += 1
    start = tuple(map(int, lines[idx].split())); idx += 1

    terminals = {}
    if lines[idx] == 'TERMINALS':
        idx += 1
        while lines[idx] != 'END':
            r, c, rew = lines[idx].split()
            terminals[(int(r), int(c))] = float(rew)
            idx += 1
        idx += 1

    walls = set()
    if lines[idx] == 'WALLS':
        idx += 1
        while lines[idx] != 'END':
            r, c = map(int, lines[idx].split())
            walls.add((r, c))
            idx += 1
        idx += 1

    step_reward = float(lines[idx]); idx += 1
    noise = float(lines[idx]); idx += 1

    return rows, cols, start, terminals, walls, step_reward, noise


def move(state, action, rows, cols, walls):
    r, c = state
    if action == 'UP': r -= 1
    elif action == 'DOWN': r += 1
    elif action == 'LEFT': c -= 1
    elif action == 'RIGHT': c += 1

    if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in walls:
        return state
    return (r, c)


def get_transitions(state, action, rows, cols, walls, noise):
    if noise == 0.0:
        return [(1.0, move(state, action, rows, cols, walls))]

    if action in ['UP', 'DOWN']:
        alt = ['LEFT', 'RIGHT']
    else:
        alt = ['UP', 'DOWN']

    return [
        (1 - noise, move(state, action, rows, cols, walls)),
        (noise / 2, move(state, alt[0], rows, cols, walls)),
        (noise / 2, move(state, alt[1], rows, cols, walls))
    ]


# ================= VALUE ITERATION =================

def value_iteration(rows, cols, terminals, walls, step_reward, noise, gamma, epsilon=1e-4):
    V = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)
            if s not in walls:
                if s in terminals:
                    V[s] = terminals[s]
                else:
                    V[s] = 0.0

    iterations = 0

    while True:
        delta = 0
        new_V = V.copy()
        iterations += 1

        for r in range(rows):
            for c in range(cols):
                s = (r, c)

                if s in walls or s in terminals:
                    continue

                best_value = -math.inf

                for action in ACTIONS:
                    q = 0
                    for prob, s2 in get_transitions(s, action, rows, cols, walls, noise):
                        reward = terminals.get(s2, step_reward)
                        q += prob * (reward + gamma * V[s2])

                    best_value = max(best_value, q)

                new_V[s] = best_value
                delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < epsilon:
            break

    print(f"Gamma = {gamma} converged in {iterations} iterations")
    return V, iterations


def extract_policy(V, rows, cols, terminals, walls, step_reward, noise, gamma):
    policy = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)

            if s in terminals or s in walls:
                continue

            best_a = None
            best_q = -math.inf

            for a in ACTIONS:
                q = 0
                for p, s2 in get_transitions(s, a, rows, cols, walls, noise):
                    reward = terminals.get(s2, step_reward)
                    q += p * (reward + gamma * V[s2])

                if q > best_q:
                    best_q = q
                    best_a = a

            policy[s] = best_a

    return policy


# ================= SAVE IMAGE =================

def save_policy_image(rows, cols, start, terminals, walls, values, policy, gamma, grid_filename):
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    for r in range(rows):
        for c in range(cols):
            x = c
            y = rows - r - 1
            s = (r, c)

            color = 'white'
            if s in walls:
                color = 'black'
            elif s in terminals:
                color = 'lightgreen' if terminals[s] > 0 else 'lightcoral'

            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

            if s == start:
                ax.text(x + 0.1, y + 0.9, 'S')

            if s in terminals:
                ax.text(x + 0.5, y + 0.5, f'{terminals[s]:.2f}', ha='center')

            if s in values and s not in terminals:
                ax.text(x + 0.5, y + 0.15, f'V={values[s]:.2f}', ha='center')

            if s in policy:
                ax.text(x + 0.5, y + 0.6, ARROWS[policy[s]], ha='center', fontsize=14)

    clean_grid = grid_filename.replace('.txt', '').replace('/', '_').replace('\\', '_')
    clean_gamma = str(gamma).replace('.', '_')
    filename = f"{clean_grid}_gamma_{clean_gamma}.png"

    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")


# ================= GUI =================

def draw_grid(rows, cols, start, terminals, walls, values, policy, title):
    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root, width=cols * CELL_SIZE, height=rows * CELL_SIZE)
    canvas.pack()

    for r in range(rows):
        for c in range(cols):
            x1 = c * CELL_SIZE
            y1 = r * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            fill = 'white'
            if (r, c) in walls:
                fill = 'black'
            elif (r, c) in terminals:
                fill = 'lightgreen' if terminals[(r, c)] > 0 else 'lightcoral'

            canvas.create_rectangle(x1, y1, x2, y2, fill=fill)

            if (r, c) in policy:
                canvas.create_text(x1 + 40, y1 + 40, text=ARROWS[policy[(r, c)]], font=('Arial', 16))

    root.mainloop()


# ================= MAIN =================

if __name__ == '__main__':
    print("Enter the gridworld spec file name:")
    grid = input().strip()

    rows, cols, start, terminals, walls, step_reward, noise = parse_gridworld(grid)

    gamma_values = [0.9, 0.95, 0.99]

    for gamma in gamma_values:
        V, iterations = value_iteration(rows, cols, terminals, walls, step_reward, noise, gamma)
        policy = extract_policy(V, rows, cols, terminals, walls, step_reward, noise, gamma)

        print(f"Gamma: {gamma}, Iterations: {iterations}")

        save_policy_image(rows, cols, start, terminals, walls, V, policy, gamma, grid)

        draw_grid(rows, cols, start, terminals, walls, V, policy, f"Gamma {gamma}")