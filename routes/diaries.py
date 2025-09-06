import os
import json
import logging
from routes import app
from flask import jsonify, request
logger = logging.getLogger(__name__)

class Sol:
    def __init__(self, tasks, edges, s0):
        self.tasks = tasks # list of dicts
        self.edges = edges # list of dicts, process to become list of 3-tuples
        self.s0 = s0
        self.INF = int(1e18)
    def process_graph(self):
        self.vertices = []
        for i, edge in enumerate(self.edges):
            u, v = edge['connection']
            self.vertices.append(u)
            self.vertices.append(v)
            
        self.vertices = list(set(self.vertices))
        for i, edge in enumerate(self.edges):
            u, v = edge['connection']
            self.edges[i] = (self.vertices.index(u), self.vertices.index(v), edge['fee'])
        self.n = len(self.vertices)
        self.calc_dists()

    def solve(self):
        self.process_graph()
        
        # TASK PROCESSING
        self.T = len(self.tasks)
        self.tasks.sort(key = lambda task: task['end'])
        self.task_names = [task['name'] for task in self.tasks]
        self.times = [0]
        for task in self.tasks:
            self.times.append(task['start'])
            self.times.append(task['end'])
        self.times = list(set(self.times))
        self.times.sort()

        for task in self.tasks: # 0-indexed
            task['station'] = self.vertices.index(task['station'])
            task['start'] = self.times.index(task['start'])
            task['end'] = self.times.index(task['end'])
        self.s0 = self.vertices.index(self.s0)

        # dp[task]
        self.dp = [(-self.INF, 0, None) for _ in range(self.T + 1)] # (score, -fee, par name)
        # 1-indexed
        self.dp[0] = (0, 0, None)
        answer = (0, 0, 0)
        for i, task in enumerate(self.tasks):
            v_sit = task['station']
            t1 = task['start']
            # print("\nprocessing", i, task['name'], task['start'], task['end'])
            for j, (score, fee, _) in enumerate(self.dp[:i + 1]):
                if j > 0:
                    t0 = self.tasks[j - 1]['end']
                    u = self.tasks[j - 1]['station']
                else:
                    t0, u = 0, self.s0
                if t0 > t1:
                    continue
                # print('     considering', t0, u, self.tasks[j - 1]['name'] if j > 0 else 'START')
                # if we take t0:
                score += task['score']
                fee -= self.d[u][v_sit]
                if (score, fee) > self.dp[i + 1][:-1]:
                    # print('update dp: ', score, fee, j)
                    self.dp[i + 1] = (score, fee, j)
                    

            score, fee, _ = self.dp[i + 1]
            fee -= self.d[v_sit][self.s0]
            # print("eval:", i + 1, task['name'], score, fee)
            if (score, fee) > answer[:-1]:
                answer = (score, fee, i + 1)
    
        logger.info(answer)
        last_task = answer[2]
        schedule = []
        fee_brute_force = 0
        score_brute_force = 0
        last_v = self.s0
        while last_task > 0:
            task_obj = self.tasks[last_task - 1]
            v = task_obj['station']
            score_brute_force += task_obj['score']

            fee_brute_force += self.d[last_v][v]
            last_v = v
            schedule.append(task_obj['name'])
            last_task = self.dp[last_task][2]
        
        fee_brute_force += self.d[last_v][self.s0]
        if not (fee_brute_force == -answer[1]):
            logger.error(f"fee incorrect: {fee_brute_force}, not {-answer[1]}")
        if not (score_brute_force == answer[0]):
            logger.error(f"score incorrect: {score_brute_force}, not {answer[0]}")


        schedule = schedule[::-1]
        answer = {
            'max_score': answer[0],
            "min_fee": -answer[1],
            "schedule": schedule
        }
        logger.info(f"N = {self.n}, T = {self.T}")
        return answer


    def calc_dists(self):
        self.d = [[self.INF] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.d[i][i] = 0
        for u, v, fee in self.edges:
            self.d[u][v] = fee
            self.d[v][u] = fee
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    self.d[i][j] = min(self.d[i][j], self.d[i][k] + self.d[k][j])

TEST_CASE = 0

@app.route("/princess-diaries", methods = ["POST"])
def princess_diaries():
    data = request.get_json(silent=True) or {}
    TEST_CASE += 1
    logger.warning(f"FULL DATA #{TEST_CASE}")
    logger.warning(data)

    # extract top-level keys
    tasks = data.get("tasks", [])
    subway = data.get("subway", [])
    starting_station = data.get("starting_station")

    # Example: log what we received
    logger.warning("Received tasks:")
    logger.warning(tasks)
    logger.warning("Received subway connections:")
    logger.warning(subway)
    logger.warning("Starting station: %s", starting_station)
    
    solution = Sol(tasks, subway, starting_station).solve()
    logger.info("Solution: %s", solution)
    
    out = {
    "max_score": int(solution.get("max_score", 0)),
    "min_fee": int(solution.get("min_fee", 0)),
    "schedule": list(solution.get("schedule", [])),
    }
    return jsonify(out)