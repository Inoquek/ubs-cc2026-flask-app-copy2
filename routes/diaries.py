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
        self.INF = int(1e9)
    def process_graph(self):
        self.vertices = []
        for i, edge in enumerate(self.edges):
            u, v = edge['connection']
            self.vertices.append(u)
            self.vertices.append(v)
            
        self.vertices = list(set(self.vertices))
        # print(f"vertices: {self.vertices}")
        for i, edge in enumerate(self.edges):
            u, v = edge['connection']
            self.edges[i] = (self.vertices.index(u), self.vertices.index(v), edge['fee'])
        self.n = len(self.vertices)
        self.calc_dists()
    
    def preprocess_tasks(self):
        for task in self.tasks:
            task['station'] = self.vertices.index(task['station'])
        self.s0 = self.vertices.index(self.s0)

    def solve(self):
        self.process_graph()
        self.preprocess_tasks()
        self.times = [0]
        for task in self.tasks:
            self.times.append(task['start'])
            self.times.append(task['end'])
        self.times = list(set(self.times))
        self.times.sort()
        # only storing ending times of tasks
        self.dp = {(0, self.s0): (0, 0, [])} # (score, -fee, schedule)
        answer = (0, 0, [])
        for task in sorted(self.tasks, key = lambda task: task['end']):
            v_sit = task['station']
            t1, t2 = self.times.index(task['start']), self.times.index(task['end'])
            self.dp[(t2, v_sit)] = (-self.INF, 0, [])
            for (t0, u), (score, fee, schedule) in self.dp.items():
                if t0 > t1:
                    continue
                # if we take t0:
                score_t0, fee_t0, schedule_t0 = self.dp[(t0, u)]
                score_t0 += task['score']
                fee_t0 -= self.d[u][v_sit]
                if (score_t0, fee_t0) >= self.dp[(t2, v_sit)][:-1]:
                    self.dp[(t2, v_sit)] = (score_t0, fee_t0, schedule_t0 + [task['name']])

            score, fee, schedule = self.dp[(t2, v_sit)]
            if_finish = (score, fee - self.d[v_sit][self.s0], schedule)
            answer = max(answer, if_finish)
        
        answer = {
            'max_score': answer[0],
            "min_fee": -answer[1],
            "schedule": answer[2]
        }
        logger.info(f"N is {self.n}, T is{len(self.times)}")
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

# INPUT = {
#   "tasks": [
#     { "name": "A", "start": 480, "end": 540, "station": 1, "score": 2 },
#     { "name": "B", "start": 600, "end": 660, "station": 2, "score": 1 },
#     { "name": "C", "start": 720, "end": 780, "station": 3, "score": 3 },
#     { "name": "D", "start": 840, "end": 900, "station": 4, "score": 1 },
#     { "name": "E", "start": 960, "end": 1020, "station": 1, "score": 4 },
#     { "name": "F", "start": 530, "end": 590, "station": 2, "score": 1 }
#   ],
#   "subway": [
#     { "connection": [0, 1], "fee": 10 },
#     { "connection": [1, 2], "fee": 10 },
#     { "connection": [2, 3], "fee": 20 },
#     { "connection": [3, 4], "fee": 30 }
#   ],
#   "starting_station": 0
# }
# tasks = INPUT['tasks']
# edges = INPUT['subway']
# s0 = INPUT['starting_station']
# print(Sol(tasks, edges, s0).solve())

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