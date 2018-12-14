import random
import numpy
import copy
import time
import sys
import getopt
import multiprocessing as mp
# random.seed(15) # 10 285 # 6 280
total_demand = 0


# 12 --- 5370，sal7:291,3721 gdb1:316 gdb10 275 sal1 :173
# 15 --- 5362, sal7: 292 ,3721 gdb10 : 275

def read_file(file_path):
    f = open(file_path, 'r')
    file_name = f.readline()
    vertices = int(f.readline().split()[2])
    deport = int(f.readline().split()[2])
    required_edges = int(f.readline().split()[3])
    none_required_edges = int(f.readline().split()[3])
    vehicles = int(f.readline().split()[2])
    capacity = int(f.readline().split()[2])
    total_coast = int(f.readline().split()[6])
    graph = []
    tasks = []
    for i in range(int(vertices + 1)):
        list = []
        for j in range(int(vertices + 1)):
            list.append(999999)
        graph.append(list)
    for i in range(int(vertices + 1)):
        graph[i][i] = 0

    for line in f:
        task_line = line.split()
        if task_line[0].isdigit():  # 含有cost和demand的任务边
            if int(task_line[3]) != 0:
                tasks.append([(int(task_line[0]), int(task_line[1])), int(task_line[2]), int(task_line[3])])

        if line.split()[0].isdigit():
            num_line = line.split()
            vertice_1 = int(num_line[0])
            vertice_2 = int(num_line[1])
            cost = int(num_line[2])
            graph[vertice_1][vertice_2] = cost
            graph[vertice_2][vertice_1] = cost
    for k in range(1, len(graph)):
        for i in range(1, len(graph)):
            for j in range(1, len(graph)):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    matrix = numpy.array(graph)
    task_matrix = numpy.array(tasks)
    return task_matrix, matrix, deport, capacity
    # print(matrix)


def initial_solution_2(tasks_matrix, graph, deport, capacity):  # 随机算法
    tasks = tasks_matrix.tolist()
    total_solution_cost = 0
    Initial_solution = []
    remain_capacity = []
    while len(tasks) != 0:
        current_point = deport
        car_capacity = capacity
        route = [0]
        while True:
            compare_cost = 100000000000
            min_task_cost = 0
            distance_cost = 0
            for task in tasks:  # 找出离当前点最小的cost
                cost = graph[current_point][task[0][0]]
                inverse_cost = graph[current_point][task[0][1]]
                if cost <= inverse_cost:
                    if cost < compare_cost:
                        compare_cost = cost
                        min_task = task
                        distance_cost = cost
                        min_task_cost = cost
                elif cost > inverse_cost:
                    if inverse_cost < compare_cost:
                        compare_cost = inverse_cost
                        min_task = [(task[0][1], task[0][0]), task[1], task[2]]
                        distance_cost = inverse_cost
                        min_task_cost = inverse_cost
            same_cost_task = []

            for task in tasks:  # 找出从当前点到该点具有
                calculate_cost = graph[current_point][task[0][0]]
                calculate_inverse_cost = graph[current_point][task[0][1]]
                if calculate_cost == distance_cost:
                    same_cost_task.append(task)
                if calculate_inverse_cost == distance_cost:
                    same_cost_task.append([(task[0][1], task[0][0]), task[1], task[2]])

            if len(same_cost_task) == 1:
                temp = min_task
            elif len(same_cost_task) > 1:  # 使用rule2找到task
                temp = random.choice(same_cost_task)  # 随机抽选结果
            final_min = temp
            if car_capacity - final_min[2] < 0:  # 满载
                route.append(0)
                total_solution_cost = total_solution_cost + graph[current_point][deport]
                break
            car_capacity -= final_min[2]
            total_solution_cost += min_task_cost + temp[1]
            route.append(final_min)
            current_point = final_min[0][1]  # 边的另一个vertice
            inverse_task = [(final_min[0][1], final_min[0][0]), final_min[1], final_min[2]]  # 当前任务的反任务边
            for task in tasks:
                if task == final_min or task == inverse_task:
                    tasks.remove(task)
                else:
                    pass
            if len(tasks) == 0:
                route.append(0)
                total_solution_cost += graph[current_point][deport]  # 所有的任务都被执行完后，加入该点到初始点deport的距离
                break
        Initial_solution.append(route)
        remain_capacity.append(car_capacity)
    solution_cost = []  # 将初始解和总的cost放到同一个集合中
    solution_cost.append(Initial_solution)
    solution_cost.append(total_solution_cost)
    solution_cost.append(remain_capacity)
    return solution_cost


def initial_solution_1(tasks_matrix, graph, deport, capacity):
    tasks = tasks_matrix.tolist()
    global total_demand
    total_solution_cost = 0
    Initial_solution = []
    remain_capacity = []
    while len(tasks) != 0:  # 当任务列表不为空时，执行
        current_point = deport
        car_capacity = capacity  # 车满或者无法再次接收demand或者任务后，跳出内部循环后，重新恢复车子的容量
        route = [0]
        while True:  # 执行 直到车子容量满或者任务结束
            compare_value = 100000000000
            min_task_cost = 0
            for task in tasks:  # 选取deport点到cost / demand 最小的任务，加入评估函数，与选取最小的cost类似
                cost = graph[current_point][task[0][0]] + task[1]  # 计算当前点到 task的第一个定点的距离
                inverse_cost = graph[current_point][task[0][1]] + task[1]  # 计算当前点到 task第二个定点的距离
                if cost > ((3 / 2) * task[2]):
                    new_comp_value = float((cost * cost) / (task[2]) + cost)
                elif cost > (2 * (task[2])):
                    new_comp_value = float((cost * cost) / (task[2]) + cost * 2)
                elif cost > (4 * (task[2])):
                    new_comp_value = float((cost * cost) / (task[2]) + cost * 4)
                elif task[2] > 3 * cost / 2:
                    new_comp_value = float(cost * cost / (task[2] * task[2]))
                elif task[2] > 2 * cost:
                    new_comp_value = float(cost * cost / (task[2] * task[2] * task[2]))
                elif task[2] > 4 * cost:  # cost 最小，相当与选取最近的
                    new_comp_value = 0
                else:
                    new_comp_value = float((cost * cost) / (task[2]) + cost)
                if inverse_cost > ((3 / 2) * (task[2])):
                    inverse_comp_value = float((inverse_cost * inverse_cost) / (task[2]) + inverse_cost)
                elif inverse_cost > (2 * (task[2])):
                    inverse_comp_value = float((inverse_cost * inverse_cost) / (task[2]) + inverse_cost * 2)
                elif inverse_cost > (4 * (task[2])):
                    inverse_comp_value = float((inverse_cost * inverse_cost) / (task[2]) + inverse_cost * 4)
                elif task[2] > 3 * inverse_cost / 2:
                    inverse_comp_value = float(inverse_cost * inverse_cost / (task[2] * task[2]))
                elif task[2] > 2 * inverse_cost:
                    inverse_comp_value = float(inverse_cost * inverse_cost / (task[2] * task[2] * task[2]))
                elif task[2] > 4 * inverse_cost:
                    inverse_comp_value = 0
                else:
                    inverse_comp_value = float((inverse_cost * inverse_cost) / (task[2]) + inverse_cost)
                if new_comp_value < inverse_comp_value:  # 正任务边与反任务边的比较 ， 选出较好的任务边去执行
                    if new_comp_value < compare_value:
                        compare_value = new_comp_value
                        min_task = task
                        min_task_cost = cost
                else:
                    if inverse_comp_value < compare_value:
                        compare_value = inverse_comp_value
                        min_task = [(task[0][1], task[0][0]), task[1], task[2]]
                        min_task_cost = inverse_cost
            if car_capacity - min_task[2] < 0:  # 满载
                route.append(0)
                total_solution_cost = total_solution_cost + graph[current_point][deport]
                break
            car_capacity -= min_task[2]
            total_demand += min_task[2]

            total_solution_cost += min_task_cost
            route.append(min_task)
            current_point = min_task[0][1]  # 边的另一个vertice
            inverse_task = [(min_task[0][1], min_task[0][0]), min_task[1], min_task[2]]  # 当前任务的反任务边
            for task in tasks:
                if task == min_task or task == inverse_task:
                    tasks.remove(task)
                else:
                    pass
            if len(tasks) == 0:
                route.append(0)
                total_solution_cost += graph[current_point][deport]  # 所有的任务都被执行完后，加入该点到初始点deport的距离
                break
        Initial_solution.append(route)
        remain_capacity.append(car_capacity)  # 车满即为一次任务结束，将车的剩余容量加入到列表中，用于后续的优化
    solution_cost = []  # 将初始解和总的cost放到同一个集合中
    solution_cost.append(Initial_solution)  # 任务顺序集合
    solution_cost.append(total_solution_cost)  # 该解的cost
    solution_cost.append(remain_capacity)  # 该解的剩余容量的list
    return solution_cost


def choose_solution(time_, tasks, graph, deport, capacity):
    start_time_ = time.time()
    solution_population = []
    solution_population.append(initial_solution_1(tasks, graph, deport, capacity))

    for i in range(1000000):
        result = initial_solution_2(tasks, graph, deport, capacity)
        solution_population.append(result)
        execute_time = time.time() - start_time_
        if execute_time > time_ * 0.3:  # 设置使用传入时间参数的1/4用生成随机初始解
            break
    sort_list = sorted(solution_population, key=lambda x: x[1])
    return sort_list


#  ---------------------------------------------遗传算法优化----------------------------------------------------------

def initial_gene_algorithm(solution, total_solution_cost, deport, remain_time, capacity, graph):
    new_solution_cost = total_solution_cost
    time_remain_limit = 2
    average_time = 0
    total_time = 0
    count_number = 0
    remain_time_ = remain_time

    while (remain_time_ > 2 * average_time) and (remain_time_ > time_remain_limit):  # 时间允许内进行交叉互换
        start_time_ = time.time()
        temp_solution = copy.deepcopy(solution)
        temp_route1 = []
        temp_route2 = []
        count = 0
        while True:
            while True:
                route_number1 = random.randint(0, len(solution) - 1)
                route_number2 = random.randint(0, len(solution) - 1)
                if (route_number1 != route_number2) and (len(solution[route_number1]) >= 4) and (
                        len(solution[route_number2]) >= 4):  # 至少为两个task在一个[]中
                    break
            task_1 = random.randint(0, len(solution[route_number1]) - 3)  # 任务序号 任务一片段1
            task_2 = random.randint(0, len(solution[route_number2]) - 3)  # 任务序号  任务二片段2
            remain_demand1 = 0
            remain_demand1_1 = 0
            remain_demand2 = 0
            remain_demand2_1 = 0
            for j in range(1, task_1 + 1):  # 左1染色体
                remain_demand1 = remain_demand1 + solution[route_number1][j][2]
            for j in range(1, task_2 + 1):  # 左2染色体
                remain_demand2 = remain_demand2 + solution[route_number2][j][2]
            for i in range(task_1 + 1, len(solution[route_number1]) - 1):  # 右1染色体
                remain_demand1_1 = remain_demand1_1 + solution[route_number1][i][2]
            for i in range(task_2 + 1, len(solution[route_number2]) - 1):  # 右2染色体
                remain_demand2_1 = remain_demand2_1 + solution[route_number2][i][2]
            if (remain_demand1 + remain_demand2_1) <= capacity and (remain_demand2 + remain_demand1_1) <= capacity:
                # 找到符合要求的解
                break
            count += 1
            if count > (remain_time_ / 6 * 50000):  # 设置迭代次数
                if new_solution_cost < total_solution_cost:
                    return solution, new_solution_cost
                else:
                    return solution, total_solution_cost

        for i in range(task_1 + 1, len(solution[route_number1])):  # 形成含有交叉片段的后半段链表
            temp_route1.append(temp_solution[route_number1][i])

        for i in range(task_2 + 1, len(solution[route_number2])):
            temp_route2.append(temp_solution[route_number2][i])

        temp_solution[route_number1] = temp_solution[route_number1][0: task_1 + 1]  # 前半段链表
        temp_solution[route_number2] = temp_solution[route_number2][0: task_2 + 1]

        temp_solution[route_number1].extend(temp_route2)
        temp_solution[route_number2].extend(temp_route1)
        mutation = random.randint(0, 100)
        if mutation > 40 and mutation < 51:
            position_1 = random.randint(1, len(temp_solution[route_number1]) - 3)
            position_2 = random.randint(1, len(temp_solution[route_number1]) - 3)
            tmp = temp_solution[route_number1][position_1]
            temp_solution[route_number1][position_1] = temp_solution[route_number1][position_2]
            temp_solution[route_number1][position_2] = tmp
        temp_cost = 0

        for route in temp_solution:
            for i in range(1, len(route) - 2):
                # print(route[i][1]," ", route[i][0][1], " ", route[i+1][0][0] )
                temp_cost = temp_cost + route[i][1] + graph[route[i][0][1]][route[i + 1][0][0]]
            temp_cost = temp_cost + graph[deport][route[1][0][0]]  # deport 点到初始点
            temp_cost = temp_cost + graph[route[len(route) - 2][0][1]][deport] + route[len(route) - 2][1]  # 终点 + 最后一个任务
        # print(temp_solution, "Temp_cost: ", temp_cost)
        if temp_cost <= new_solution_cost:  # 如果找到新的好解，解交换
            new_solution_cost = temp_cost
            solution = temp_solution
        count_number = count_number + 1
        once_time = time.time() - start_time_  # 得到一个cost较小的新route片段
        total_time = total_time + once_time
        average_time = total_time / count_number  # 计算到目前为止的平均时间
        remain_time_ = remain_time_ - once_time
        # print("Remain_time:", remain_time_, "Total_time:", total_time)

    if new_solution_cost < total_solution_cost:
        return solution, new_solution_cost
    else:
        return solution, total_solution_cost


# ----------------------------------------------测试区域----------------------------------------------------------------------
# egl-s1-A.dat
# sal7A.dat
# gdb1.dat
# egl-s1-A.dat
class Worker(mp.Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
    def run(self):
        while True:
            task_seed = self.inQ.get()  # 取出任务， 如果队列为空， 这一步会阻塞直到队列有元素
            a, b = proccess_algorithm(task_seed)  # 执行任务
            self.outQ.put((a, b))  # 返回结果

def create_worker (num):
    '''
    创建子进程备用
    :param num: 多线程数量
    '''
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue()))
        worker[i].start()


def finish_worker ():
    '''
    关闭所有子线程
    '''
    for w in worker:
        w.terminate()


def get_paramater():
    result = []
    _, args = getopt.getopt(sys.argv[0:2], "")
    opts, _ = getopt.getopt(sys.argv[2:], "t:s:")
    result.append(args[1])
    for i in opts:
        result.append(i[1])
    return result

def proccess_algorithm(seed_):
    parameter = get_paramater()
    file_path = parameter[0]
    limit_time = int(parameter[1])
    random.seed(seed_)
    start_time = time.time()  # 开始时间
    tasks, graph, deport, capacity = read_file(file_path)
    sort_list = choose_solution(limit_time, tasks, graph, deport, capacity)
    execute_time = time.time() - start_time  # 找到最初解的时间
    remain_time = limit_time - execute_time  # 剩余时间
    a, b = initial_gene_algorithm(sort_list[0][0], sort_list[0][1], deport, remain_time, capacity, graph)
    return a, b


if __name__ == "__main__":
    worker = []
    worker_num = 8
    create_worker(worker_num)
    seed = [15, 6, 10, random.randint(11, 999), random.randint(1000,5000), random.randint(5001, 9999), random.randint(10000, 50000), random.randint(50000, 999999)]
    for i, t in enumerate(seed):
        worker[i].inQ.put(t)
    result = []
    for i, t in enumerate(seed):
        result.append(worker[i].outQ.get())
    sort_list = sorted(result, key=lambda x: x[1])
    str1 = "s "
    a = sort_list[0][0]
    b = sort_list[0][1]
    for i in a:
        for element in i:
            if element == 0:

                str1 = str1 + "0,"
            else:
                str1 = str1 + "(" + str(element[0][0]) + "," + str(element[0][1]) + "),"
    print(str1[:-1])
    print("q ", b)
    finish_worker()

end = time.time()
