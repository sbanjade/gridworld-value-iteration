## DO NOT MAKE ANY CHANGES TO THE CODE UNLESS INSTRUCTED. ONLY MODULES STATED IN THE INSTRUCTION CAN BE IMPORTED
import tkinter as tk
import math

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}
CELL_SIZE = 80
FONT_VALUE = ('Arial', 10)
FONT_LABEL = ('Arial', 16, 'bold')


def parse_gridworld(filename):
    # This function will extract all the information from the gridworld spec and design the gridworld.
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    idx = 0
    rows, cols = map(int, lines[idx].split())
    idx += 1
    start = tuple(map(int, lines[idx].split()))
    idx += 1

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

    step_reward = float(lines[idx])
    idx += 1
    noise = float(lines[idx])
    idx += 1

    return rows, cols, start, terminals, walls, step_reward, noise


def move(state, action, rows, cols, walls):
    r, c = state

    if action == 'UP':
        r -= 1
    if action == 'DOWN':
        r += 1
    if action == 'LEFT':
        c -= 1
    if action == 'RIGHT':
        c += 1

    if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in walls:
        return state
    return (r, c)


def get_transitions(state, action, rows, cols, walls, noise):
    """Returns list of (probability, next_state)"""
    if noise == 0.0:
        return [(1.0, move(state, action, rows, cols, walls))]

    if action in ['UP', 'DOWN']:
        alt = ['LEFT', 'RIGHT']
    else:
        alt = ['UP', 'DOWN']

    transitions = []
    transitions.append((1 - noise, move(state, action, rows, cols, walls)))
    transitions.append((noise / 2, move(state, alt[0], rows, cols, walls)))
    transitions.append((noise / 2, move(state, alt[1], rows, cols, walls)))
    return transitions


# ==================================================
# Value Iteration
# ==================================================

def value_iteration(rows, cols, terminals, walls, step_reward, noise, gamma, epsilon=1e-4):
    # Implement the Value Iteration Algorithm to generate the Values for each state. Use epsilon to check for convergence

    V = {}

    # Initialize values
    for r in range(rows):
        for c in range(cols):
            s = (r, c)
            if s in walls:
                continue
            if s in terminals:
                V[s] = terminals[s]
            else:
                V[s] = 0.0

    iteration = 0

    while True:
        delta = 0
        new_V = V.copy()
        iteration += 1

        for r in range(rows):
            for c in range(cols):
                s = (r, c)

                if s in walls:
                    continue

                if s in terminals:
                    new_V[s] = terminals[s]
                    continue

                best_value = -math.inf

                for action in ACTIONS:
                    q_value = 0

                    for p, s2 in get_transitions(s, action, rows, cols, walls, noise):
                        reward = terminals.get(s2, step_reward)
                        q_value += p * (reward + gamma * V[s2])

                    if q_value > best_value:
                        best_value = q_value

                new_V[s] = best_value
                delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < epsilon:
            break

    print(f"Gamma = {gamma} converged in {iteration} iterations")
    return V


def init_values(rows, cols, walls, terminals):
    """Initialize all state values to 0.0 except walls."""
    values = {}
    pol = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)
            if s not in walls:
                if s in terminals:
                    values[s] = terminals[s]
                else:
                    values[s] = 0.0

                if s not in terminals:
                    pol[s] = 'UP'

    return values, pol


# ==================================================
# Policy Extraction
# ==================================================

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


def draw_grid(rows, cols, start, terminals, walls, values, policy, title='Grid World with State Values'):
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

            canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline='gray')

            # Start state
            if (r, c) == start:
                canvas.create_text(x1 + CELL_SIZE / 2, y1 + 15, text='S', font=FONT_LABEL)

            # Terminal reward
            if (r, c) in terminals:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE / 2,
                    text=f'{terminals[(r, c)]:.2f}',
                    font=FONT_VALUE
                )

            # State value
            if (r, c) in values and (r, c) not in terminals:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE - 15,
                    text=f'V={values[(r, c)]:.2f}',
                    font=FONT_VALUE
                )

            # Policy arrow
            if (r, c) in policy:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE / 2,
                    text=ARROWS[policy[(r, c)]],
                    font=('Arial', 18)
                )

    root.mainloop()


if __name__ == '__main__':
    print("Enter the gridworld spec")
    grid = input().strip()

    rows, cols, start, terminals, walls, step_reward, noise = parse_gridworld(grid)

    values, pol = init_values(rows, cols, walls, terminals)
    draw_grid(rows, cols, start, terminals, walls, values, pol, title='Initial Grid World')

    gamma_values = [0.9, 0.95, 0.99]

    for discount in gamma_values:
        V = value_iteration(rows, cols, terminals, walls, step_reward, noise, discount)
        policy = extract_policy(V, rows, cols, terminals, walls, step_reward, noise, discount)
        draw_grid(
            rows,
            cols,
            start,
            terminals,
            walls,
            V,
            policy,
            title=f'Final Policy (gamma = {discount})'
        )