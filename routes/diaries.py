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
        
        for task in self.tasks:
            task['station'] = self.vertices.index(task['station'])
        self.s0 = self.vertices.index(self.s0)

        self.name_to_task = {}
        self.times = [0]
        for task in self.tasks:
            self.name_to_task[task['name']] = task
            self.times.append(task['start'])
            self.times.append(task['end'])
        self.times = list(set(self.times))
        self.times.sort()
        # only storing ending times of tasks
        self.dp = {(0, self.s0): (0, 0, None)} # (score, -fee, name)
        answer = (0, 0, None)
        self.par = {}
        for task in sorted(self.tasks, key = lambda task: task['end']):
            v_sit = task['station']
            t1, t2 = self.times.index(task['start']), self.times.index(task['end'])
            self.dp[(t2, v_sit)] = (-self.INF, 0, [])
            for (t0, u), (score, fee, name) in self.dp.items():
                if t0 > t1:
                    continue
                # if we take t0:
                score += task['score']
                fee -= self.d[u][v_sit]
                if (score, fee) > self.dp[(t2, v_sit)][:-1]:
                    self.dp[(t2, v_sit)] = (score, fee, task['name'])
                    self.par[task['name']] = name

            score, fee, schedule = self.dp[(t2, v_sit)]
            fee -= self.d[v_sit][self.s0]
            if (score, fee) > answer[:-1]:
                answer = (score, fee, task['name'])
        
        logger.info(answer)
        last_task = answer[2]
        schedule = []
        fee_brute_force = 0
        score_brute_force = 0
        last_v = self.s0
        while last_task is not None:
            task_obj = self.name_to_task[last_task]
            v = task_obj['station']
            score_brute_force += task_obj['score']

            fee_brute_force += self.d[last_v][v]
            last_v = v
            schedule.append(last_task)
            last_task = self.par[last_task]
        
        fee_brute_force += self.d[last_v][self.s0]
        if not (fee_brute_force == -answer[1]):
            logger.error(f"fee incorrect: {fee_brute_force}, not {-answer[1]}")
        if not (score_brute_force == answer[0]):
            logger.error(f"fee incorrect: {score_brute_force}, not {answer[0]}")


        schedule = schedule[::-1]
        answer = {
            'max_score': answer[0],
            "min_fee": -answer[1],
            "schedule": schedule
        }
        logger.info(f"N = {self.n}, T = {len(self.tasks)}")
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


@app.route("/princess-diaries", methods = ["POST"])
def princess_diaries():
    data = request.get_json(silent=True) or {}

    # extract top-level keys
    tasks = data.get("tasks", [])
    subway = data.get("subway", [])
    starting_station = data.get("starting_station")

    # Example: log what we received
    logger.info("Received %d tasks", len(tasks))
    logger.info("Received %d subway connections", len(subway))
    logger.info("Starting station: %s", starting_station)
    
    solution = Sol(tasks, subway, starting_station).solve()
    logger.info("Solution: %s", solution)
    
    out = {
    "max_score": int(solution.get("max_score", 0)),
    "min_fee": int(solution.get("min_fee", 0)),
    "schedule": list(solution.get("schedule", [])),
    }
    return jsonify(out)