import heapq
from collections import defaultdict
import math
import json
import logging
from routes import app
from flask import jsonify, request
logger = logging.getLogger(__name__)

class MicroMousePathfinder:
    def __init__(self, horizontal_borders, vertical_borders):
        """
        Initialize the pathfinder with maze borders.
        
        Args:
            horizontal_borders: Set of (x,y) tuples representing horizontal walls
                               where (x,y) is the cell below the wall
            vertical_borders: Set of (x,y) tuples representing vertical walls
                             where (x,y) is the cell to the left of the wall
        """
        self.horizontal_borders = set(horizontal_borders)
        self.vertical_borders = set(vertical_borders)
        
        # Direction mappings (0=North, 1=NE, 2=East, 3=SE, 4=South, 5=SW, 6=West, 7=NW)
        self.directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.dir_deltas = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        # Movement base times
        self.BASE_TIMES = {
            'in_place_turn': 200,
            'default_rest': 200,
            'half_step_cardinal': 500,
            'half_step_intercardinal': 600,
            'corner_tight': 700,
            'corner_wide': 1400
        }
        
        # Momentum reduction table
        self.momentum_reduction = {
            0.0: 0.00, 0.5: 0.10, 1.0: 0.20, 1.5: 0.275,
            2.0: 0.35, 2.5: 0.40, 3.0: 0.45, 3.5: 0.475, 4.0: 0.50
        }

    def get_reduction_factor(self, m_eff):
        """Get momentum reduction factor for given effective momentum."""
        m_eff = max(0.0, min(4.0, m_eff))
        if m_eff in self.momentum_reduction:
            return self.momentum_reduction[m_eff]
        
        # Linear interpolation for values between table entries
        keys = sorted(self.momentum_reduction.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= m_eff <= keys[i + 1]:
                x1, y1 = keys[i], self.momentum_reduction[keys[i]]
                x2, y2 = keys[i + 1], self.momentum_reduction[keys[i + 1]]
                return y1 + (y2 - y1) * (m_eff - x1) / (x2 - x1)
        return 0.0

    def is_wall_between(self, x1, y1, x2, y2):
        """Check if there's a wall between two adjacent half-step positions."""
        
        if x2 % 2 and y2 % 2:
            # Corner point
            wall_r = y2 // 2
            wall_c = x2 // 2

            if ((wall_c, wall_r) in self.vertical_borders) or (wall_c, wall_r + 1) in self.vertical_borders:
                return True
            

            if ((wall_c, wall_r) in self.horizontal_borders) or (wall_c + 1, wall_r) in self.horizontal_borders:
                return True
        elif x2 % 2:
            # Can belong to vertical wall
            wall_r, wall_c = y2 // 2, x2 // 2
            if (wall_c, wall_r) in self.vertical_borders:
                return True
        elif y2 % 2:
            wall_r, wall_c = y2 // 2, x2 // 2
            if (wall_c, wall_r) in self.horizontal_borders:
                return True

        return False

    def get_possible_moves(self, x, y, direction, momentum):
        """Get all possible moves from current state."""
        moves = []
        
        # In-place rotations (only at momentum 0)
        if momentum == 0:
            moves.append({
                'command': 'L',
                'new_x': x, 'new_y': y,
                'new_dir': (direction - 1) % 8,
                'new_momentum': 0,
                'time': self.BASE_TIMES['in_place_turn']
            })
            moves.append({
                'command': 'R',
                'new_x': x, 'new_y': y,
                'new_dir': (direction + 1) % 8,
                'new_momentum': 0,
                'time': self.BASE_TIMES['in_place_turn']
            })
        
        # Longitudinal movements
        for move_type in ['F0', 'F1', 'F2', 'BB']:
            new_momentum = self.apply_momentum_change(momentum, move_type)
            if new_momentum is None:  # Invalid move
                continue
            
            # Calculate movement
            m_eff = (abs(momentum) + abs(new_momentum)) / 2.0
            
            # Determine if we actually move
            if momentum != 0 or move_type != 'BB':
                # Calculate new position
                dx, dy = self.dir_deltas[direction]
                new_x = x + dx
                new_y = y + dy
                
                # Check bounds (0 to 30 for half-step grid in 16x16 maze)
                if not (0 <= new_x <= 30 and 0 <= new_y <= 30):
                    continue
                
                # Check for walls
                if self.is_wall_between(x, y, new_x, new_y):
                    continue
                
                # Calculate time
                is_cardinal = direction % 2 == 0
                base_time = self.BASE_TIMES['half_step_cardinal'] if is_cardinal else self.BASE_TIMES['half_step_intercardinal']
                reduction = self.get_reduction_factor(m_eff)
                time = int(base_time * (1 - reduction))
                
            else:
                # No movement 
                new_x, new_y = x, y
                time = self.BASE_TIMES['default_rest']
            
            moves.append({
                'command': move_type,
                'new_x': new_x, 'new_y': new_y,
                'new_dir': direction,
                'new_momentum': new_momentum,
                'time': time
            })
            
            # Moving rotations (only if m_eff <= 1)
            if new_x != x or new_y != y:  # Only if we actually moved
                if m_eff <= 1:
                    for rot in ['L', 'R']:
                        rot_dir = 1 if rot == 'R' else -1
                        moves.append({
                            'command': move_type + rot,
                            'new_x': new_x, 'new_y': new_y,
                            'new_dir': (direction + rot_dir) % 8,
                            'new_momentum': new_momentum,
                            'time': time  # Same time as base move
                        })
        
        return moves

    def apply_momentum_change(self, current_momentum, move_type):
        """Apply momentum change based on move type."""
        if move_type == 'F0':
            return max(0, current_momentum - 1) if current_momentum >= 0 else None
        elif move_type == 'F1':
            return current_momentum if current_momentum >= 0 else None
        elif move_type == 'F2':
            return min(4, current_momentum + 1) if current_momentum >= 0 else None
        elif move_type == 'BB':
            if current_momentum > 0:
                return max(0, current_momentum - 2)
            elif current_momentum < 0:
                return min(0, current_momentum + 2)
            else:
                return 0
        return None

    def find_shortest_paths(self, start_x, start_y, start_direction, start_momentum):
        """
        Find shortest paths to all reachable cells using Dijkstra's algorithm.
        
        Args:
            start_x, start_y: Starting position in half-step coordinates (0-30)
            start_direction: Starting direction (0-7)
            start_momentum: Starting momentum (-4 to 4)
        
        Returns:
            Dictionary with (x, y, dir, momentum) as keys and (min_time, command_sequence) as values
        """
        # State: (x, y, direction, momentum)
        start_state = (start_x, start_y, start_direction, start_momentum)
        
        # Priority queue: (time, state)
        pq = [(0, start_state)]
        
        # Distance and path tracking
        distances = {start_state: 0}
        paths = {start_state: []}
        
        # Result dictionary: (x, y) -> (min_time, commands)
        results = defaultdict(lambda: (float('inf'), []))
        results[(start_x, start_y, start_direction, start_momentum)] = (0, [])
        
        visited = set()
        
        while pq:
            current_time, current_state = heapq.heappop(pq)
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            x, y, direction, momentum = current_state
            
            # Update result if this is a better path to this cell
            if current_time < results[current_state][0]:
                results[current_state] = (current_time, paths[current_state][:])
            
            # Get possible moves
            possible_moves = self.get_possible_moves(x, y, direction, momentum)
            
            for move in possible_moves:
                new_state = (move['new_x'], move['new_y'], move['new_dir'], move['new_momentum'])
                new_time = current_time + move['time']  # Add thinking time
                
                if new_state not in distances or new_time < distances[new_state]:
                    distances[new_state] = new_time
                    paths[new_state] = paths[current_state] + [move['command']]
                    heapq.heappush(pq, (new_time, new_state))
        
        # Convert to cell coordinates and return
        cell_results = {}
        for (hx, hy, dir, momentum), (time, commands) in results.items():
            if hx % 2 or hy % 2:
                continue
            cell_x, cell_y = hx // 2, hy // 2
            if (cell_x, cell_y, dir, momentum) not in cell_results or time < cell_results[(cell_x, cell_y, dir, momentum)][0]:
                cell_results[(cell_x, cell_y, dir, momentum)] = (time, commands)
        
        return cell_results

N = 16  
# start is 0, 0, goal is (7-8, 7-8)
MOVE_FOR = ["F2", "F0"]
# "R" or "L" to turn 45deg
TURN_TIME = 200
STR_TIME = 950
DIAG_TIME = 1150
DIRS = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

def rotate_dir(dir, amt):
    return DIRS[(DIRS.index(dir) + amt + 8) % 8]

def is_diag(dir):
    return abs(dir[0]) + abs(dir[1]) == 2

class Sol:
    def __init__(self):
        self.game_uuid = None
        self.v_borders = [[None for _ in range(N)] for _ in range(N - 1)]  # right of (i, j)
        self.h_borders = [[None for _ in range(N - 1)] for _ in range(N)]  # top of (i, j)
        # we also have datapts of diag direction being obstructed
        self.obstructed_diags = [[None for _ in range(N - 1)] for _ in range(N - 1)]
        self.cur_dir = [0, 1]
        self.cur_pos = [0, 0]
        self.visited = set()  # Track visited positions during exploration
        self.move_count = 0
        self.exploring = True
        self.path_finder = None
        self.finished = False
        
    def h(self, node):
        """
        Heuristic function for A* that estimates time to reach goal from node.
        node = (pos, dir) where pos = [x, y] and dir is one of DIRS
        Goal cells are (7,7), (7,8), (8,7), (8,8)
        """
        pos, dir = node
        x, y = pos
        
        # Check if we're already at goal
        if 7 <= x <= 8 and 7 <= y <= 8:
            return 0
        
        # Calculate distance to nearest goal cell
        goal_cells = [(7, 7), (7, 8), (8, 7), (8, 8)]
        min_time = float('inf')
        
        for gx, gy in goal_cells:
            dx = abs(x - gx)
            dy = abs(y - gy)
            
            # Optimal movement: use diagonal moves to reduce distance efficiently
            diag_moves = min(dx, dy)
            straight_moves = abs(dx - dy)
            
            # Base movement time (assuming no obstacles)
            move_time = diag_moves * DIAG_TIME + straight_moves * STR_TIME
            
            # Estimate turning cost
            if dx == 0 and dy == 0:
                turn_cost = 0
            else:
                # Direction vector to goal (normalized)
                target_dx = 1 if gx > x else (-1 if gx < x else 0)
                target_dy = 1 if gy > y else (-1 if gy < y else 0)
                target_dir = [target_dx, target_dy]
                
                # Find minimum turns needed
                min_turns = float('inf')
                for i, d in enumerate(DIRS):
                    if d == target_dir:
                        cur_idx = DIRS.index(dir)
                        turns = min(abs(i - cur_idx), 8 - abs(i - cur_idx))
                        min_turns = min(min_turns, turns)
                
                if min_turns == float('inf'):
                    # Approximate turn cost if exact match not found
                    min_turns = 1
                
                turn_cost = min_turns * TURN_TIME
            
            total_time = move_time + turn_cost
            min_time = min(min_time, total_time)
        
        return min_time
    
    def add_border_info(self, cell1, cell2, sensor):
        """Add border information between two adjacent cells."""
        # Check bounds
        if (min(cell1[0], cell2[0]) < 0 or max(cell1[0], cell2[0]) >= N or 
            min(cell1[1], cell2[1]) < 0 or max(cell1[1], cell2[1]) >= N):
            return
        
        # Vertical border (different x coordinates)
        if cell1[1] == cell2[1] and abs(cell1[0] - cell2[0]) == 1:
            border_x = min(cell1[0], cell2[0])
            if 0 <= border_x < N - 1:
                if self.v_borders[border_x][cell1[1]] is None:
                    logger.info(f"discovered new v_border: {border_x, cell1[1]} {'exists' if sensor else 'doesnt exist'}")
                self.v_borders[border_x][cell1[1]] = sensor
        # Horizontal border (different y coordinates)
        elif cell1[0] == cell2[0] and abs(cell1[1] - cell2[1]) == 1:
            border_y = min(cell1[1], cell2[1])
            if 0 <= border_y < N - 1:
                if self.h_borders[cell1[0]][border_y] is None:
                    logger.info(f"discovered new h_border:{cell1[0], border_y} {'exists' if sensor else 'doesnt exist'}")
                self.h_borders[cell1[0]][border_y] = sensor
    
    def can_move(self, from_pos, to_pos):
        """
        Check if movement from from_pos to to_pos is possible based on known borders.
        Returns: True if free, False if blocked, None if unknown
        """
        fx, fy = from_pos
        tx, ty = to_pos
        
        # Check bounds
        if not (0 <= tx < N and 0 <= ty < N):
            return False
        
        dx = tx - fx
        dy = ty - fy
        
        # Straight moves
        if abs(dx) + abs(dy) == 1:
            if dx == 1:  # Moving right
                if fx < N - 1:
                    border = self.v_borders[fx][fy]
                    return None if border is None else (border == 0)
            elif dx == -1:  # Moving left
                if tx < N - 1:
                    border = self.v_borders[tx][ty]
                    return None if border is None else (border == 0)
            elif dy == 1:  # Moving up
                if fy < N - 1:
                    border = self.h_borders[fx][fy]
                    return None if border is None else (border == 0)
            elif dy == -1:  # Moving down
                if ty < N - 1:
                    border = self.h_borders[tx][ty]
                    return None if border is None else (border == 0)
        
        # Diagonal moves
        elif abs(dx) == 1 and abs(dy) == 1:
            diag_x = min(fx, tx)
            diag_y = min(fy, ty)
            
            # Check diagonal obstruction
            if 0 <= diag_x < N - 1 and 0 <= diag_y < N - 1:
                diag_status = self.obstructed_diags[diag_x][diag_y]
                if diag_status == 1:  # Known to be obstructed
                    return False
                elif diag_status == 0:  # Known to be free
                    return True
                # If None, check adjacent borders
            
            # Check if adjacent straight moves are blocked
            # For diagonal move, we need both adjacent borders to be free
            h_free = self.can_move(from_pos, [fx, ty])  # Horizontal component
            v_free = self.can_move(from_pos, [tx, fy])  # Vertical component
            
            if h_free is False or v_free is False:
                return False
            if h_free is None or v_free is None:
                return None  # Unknown
            
            return True  # Both adjacent moves are free
        
        return False  # Invalid move
    
    def get_valid_moves(self, pos, dir):
        """
        Get all valid moves from current position and direction.
        Returns list of (new_pos, new_dir, cost, action) tuples.
        """
        moves = []
        
        # Turn left
        new_dir = rotate_dir(dir, -1)
        moves.append((pos, new_dir, TURN_TIME, "L"))
        
        # Turn right
        new_dir = rotate_dir(dir, 1)
        moves.append((pos, new_dir, TURN_TIME, "R"))
        
        # Move forward
        new_pos = [pos[0] + dir[0], pos[1] + dir[1]]
        can_move = self.can_move(pos, new_pos)
        
        if can_move is not False:  # Either True or None (unknown)
            move_cost = DIAG_TIME if is_diag(dir) else STR_TIME
            # If unknown, add a small penalty to encourage exploring known paths first
            if can_move is None:
                move_cost += 50
            moves.append((new_pos, dir, move_cost, "FORWARD"))
        
        return moves
    
    def find_path_to_goal(self, max_steps=1000):
        """
        Use A* to find path to goal, considering current knowledge.
        Returns path as list of actions, or None if no path found.
        """
        start_state = (tuple(self.cur_pos), tuple(self.cur_dir))
        
        # Priority queue: (f_score, g_score, state, path)
        pq = [(self.h((self.cur_pos, self.cur_dir)), 0, start_state, [])]
        g_scores = {start_state: 0}
        closed_set = set()
        
        steps = 0
        while pq and steps < max_steps:
            steps += 1
            f_score, g_score, state, path = heapq.heappop(pq)
            
            if state in closed_set:
                continue
            closed_set.add(state)
            
            pos, dir = list(state[0]), list(state[1])
            
            # Check if we reached goal
            if 7 <= pos[0] <= 8 and 7 <= pos[1] <= 8:
                return path
            
            # Explore neighbors
            for new_pos, new_dir, cost, action in self.get_valid_moves(pos, dir):
                new_state = (tuple(new_pos), tuple(new_dir))
                new_g = g_score + cost
                
                if new_state in closed_set:
                    continue
                
                if new_state not in g_scores or new_g < g_scores[new_state]:
                    g_scores[new_state] = new_g
                    h_val = self.h((new_pos, new_dir))
                    new_f = new_g + h_val
                    new_path = path + [action]
                    heapq.heappush(pq, (new_f, new_g, new_state, new_path))
        
        return None
    
    def find_exploration_target(self):
        """
        Find the best direction to explore for gathering information.
        Returns path to nearest unexplored area or unknown border.
        """
        start_state = (tuple(self.cur_pos), tuple(self.cur_dir))
        
        # BFS to find nearest unexplored cell or unknown border
        from collections import deque
        queue = deque([(start_state, [])])
        visited = {start_state}
        
        while queue:
            state, path = queue.popleft()
            pos, dir = list(state[0]), list(state[1])
            
            # Check if this position gives us new information
            if (tuple(pos) not in self.visited and tuple(pos) != tuple(self.cur_pos)) or \
               self.has_unknown_borders(pos):
                return path
            
            # If path is getting too long, prioritize shorter explorations
            if len(path) > 20:
                continue
            
            for new_pos, new_dir, cost, action in self.get_valid_moves(pos, dir):
                new_state = (tuple(new_pos), tuple(new_dir))
                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [action]
                    queue.append((new_state, new_path))
        
        # If no unexplored area found, return empty path
        return []
    
    def has_unknown_borders(self, pos):
        """Check if position has unknown borders that could give us information."""
        x, y = pos
        for dx, dy in DIRS:
            adj_pos = [x + dx, y + dy]
            if 0 <= adj_pos[0] < N and 0 <= adj_pos[1] < N:
                if self.can_move(pos, adj_pos) is None:
                    return True
        return False
    
    def init_path_finder(self):
        # just started exploiting, let's take worst case scenario of each unknown border
        v_borders_set = set()
        h_borders_set = set()
        for i in range(N - 1):
            for j, x in enumerate(self.v_borders[i]):
                if x != 0:
                    v_borders_set.add((i, j))
        
        for i in range(N):
            for j, x in enumerate(self.h_borders[i]):
                if x != 0:
                    h_borders_set.add((i, j))

        self.path_finder = MicroMousePathfinder(h_borders_set, v_borders_set)
        self.exploit_path = self.path_finder.find_shortest_paths(self.cur_pos[0] * 2, self.cur_pos[1] * 2, DIRS.index(self.cur_dir), 0)
        self.exploit_path = self.exploit_path[(0, 0, 0, 0)][1]
        self.cur_dir = [0, 1]
        self.cur_pos = [0, 0]

    def is_goal(self, pos):
        return min(pos) >= 7 and max(pos) <= 8

    def next_instructions(self):
        if self.exploring:
            # Try to find direct path to goal first
            path_to_goal = self.find_path_to_goal()
            
            if path_to_goal:
                # We have a path to goal, execute first move(s)
                return self.convert_path_to_moves(path_to_goal[:3])  # Execute up to 3 moves
            else:
                # No clear path to goal, explore
                exploration_path = self.find_exploration_target()
                if exploration_path:
                    return self.convert_path_to_moves(exploration_path[:3])  # Execute up to 3 moves
            
            # Fallback: try to move in any valid direction
            valid_moves = self.get_valid_moves(self.cur_pos, self.cur_dir)
            
            for new_pos, new_dir, cost, action in valid_moves:
                if action == "FORWARD" and self.can_move(self.cur_pos, new_pos) is not False:
                    return MOVE_FOR
                elif action in ["L", "R"]:
                    self.cur_dir = new_dir
                    return [action]
        else:
            if self.path_finder is None:
                self.init_path_finder()
            elif not self.finished:
                exploit_paths = self.path_finder.find_shortest_paths(self.cur_pos[0] * 2, self.cur_pos[1] * 2, DIRS.index(self.cur_dir), 0)
                best_exploit = (1e9,)
                for target in [(7, 7, 0, 0), (7, 8, 0, 0), (8, 7, 0, 0), (8, 8, 0, 0)]:
                    best_exploit = min(best_exploit, exploit_paths.get(target, (1e9,)))
                self.exploit_path = best_exploit[1]
                self.finished = True
            else:
                self.exploit_path = []
            return self.exploit_path
                

    def new_request(self, req):
        """
        Process new sensor information and return next moves to execute.
        Returns list of moves like ["L"], ["R"], or ["R2", "R0"].
        """
        if self.game_uuid is None:
            self.game_uuid = req['game_uuid']
        assert self.game_uuid == req['game_uuid']
        
        # Mark current position as visited
        self.visited.add(tuple(self.cur_pos))
        self.move_count += 1
        
        # Update border knowledge from sensor data
        for amt in range(-2, 3):  # Fixed range to include all 5 sensors
            new_dir = rotate_dir(self.cur_dir, amt)
            sensor = req['sensor_data'][amt + 2]
            
            if not is_diag(new_dir):
                # Straight direction sensor
                new_pos = [self.cur_pos[0] + new_dir[0], self.cur_pos[1] + new_dir[1]]
                self.add_border_info(self.cur_pos, new_pos, sensor)
            else:
                # Diagonal direction sensor
                new_diag_pos = [self.cur_pos[0] + new_dir[0], self.cur_pos[1] + new_dir[1]]
                diag_x = min(self.cur_pos[0], new_diag_pos[0])
                diag_y = min(self.cur_pos[1], new_diag_pos[1])
                
                if 0 <= diag_x < N - 1 and 0 <= diag_y < N - 1:
                    if sensor:
                        if self.obstructed_diags[diag_x][diag_y] is None:
                            logger.info(f"discovered obstructed diag: {diag_x, diag_y}")
                        self.obstructed_diags[diag_x][diag_y] = 1
                    else:
                        self.obstructed_diags[diag_x][diag_y] = 0
                        # If diagonal is free, adjacent borders are also free
                        adjacent_positions = [
                            ([diag_x, diag_y], [diag_x + 1, diag_y]),
                            ([diag_x, diag_y], [diag_x, diag_y + 1]),
                            ([diag_x + 1, diag_y + 1], [diag_x, diag_y + 1]),
                            ([diag_x + 1, diag_y + 1], [diag_x + 1, diag_y])
                        ]
                        for pos1, pos2 in adjacent_positions:
                            self.add_border_info(pos1, pos2, 0)
        
        logger.info(f"valid_moves: {self.get_valid_moves(self.cur_pos, self.cur_dir)}")

        if self.exploring and any([self.is_goal(valid_move[0]) for valid_move in self.get_valid_moves(self.cur_pos, self.cur_dir)]):
            self.exploring = False

        instructions = self.next_instructions()
        move = {'instructions': instructions, 'end': self.finished}
        logger.info(move)

        
        return move
    
    def convert_path_to_moves(self, path):
        """
        Convert a path of actions to actual move commands.
        Updates cur_pos and cur_dir as moves are planned.
        """
        if not path:
            return []
        
        moves = []
        temp_pos = self.cur_pos[:]
        temp_dir = self.cur_dir[:]
        
        logger.info(path)

        for action in path:
            if action == "L":
                temp_dir = rotate_dir(temp_dir, -1)
                moves.append("L")
                # Update actual state for turn
                self.cur_dir = temp_dir
            elif action == "R":
                temp_dir = rotate_dir(temp_dir, 1)
                moves.append("R")
                # Update actual state for turn
                self.cur_dir = temp_dir
            elif action == "FORWARD":
                new_pos = [temp_pos[0] + temp_dir[0], temp_pos[1] + temp_dir[1]]
                # Check if move is still valid (border info might have changed)
                if self.can_move(temp_pos, new_pos) is not False:
                    temp_pos = new_pos
                    moves.extend(MOVE_FOR)
                    # Update actual state for forward move
                    self.cur_pos = new_pos
                    break  # Only one forward move per call
                else:
                    break  # Can't move forward, stop here
        
        return moves
    
id_to_sol = {}

# TEST_CASE = 0
@app.route("/micro-mouse", methods = ["POST"])
def mouse():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    if data['uuid'] not in id_to_sol:
        sol = Sol()
        id_to_sol[data['uuid']] = sol
    else:
        sol = id_to_sol[data['uuid']]
    # TEST_CASE += 1
    logger.info(f"---PROCESSING QUERY with uuid={data['uuid']}---")
    logger.info(data)

    answer = sol.new_request()

    return jsonify(answer)