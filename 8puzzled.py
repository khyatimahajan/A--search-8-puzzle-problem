import numpy as np
import math

explored_nodes = 0
expanded_nodes = 0

def get_goal_state():
	''' Helper function for goal state information '''
	print("Please enter the goal state configuration in this format- example: 1 2 3 4 5 6 7 8 0.")
	take_input = input("Enter goal state configuration now:\n").split()
	goal_state = np.asarray(take_input, dtype=np.int)
	return goal_state

def get_initial_state():
	''' Helper function for initial state information '''
	print("Please enter the initial state configuration in this format- example: 1 2 3 4 5 6 7 8 0.")
	take_input = input("Enter initial state configuration now:\n").split()
	initial_state = np.asarray(take_input, dtype=np.int)
	return initial_state

class Node():
	''' Class for building the tree with respect to A* search '''

	# Generator function for node properties
	def __init__(self, state, action, g_cost, h_cost, parent):
		self.state = state
		self.action = action
		self.g_cost = g_cost
		self.h_cost = h_cost
		self.parent = parent
		self.child_moves = {"up": None, "left": None, "right": None, "down": None}
	
	# Can we move the blank tile up?
	def can_move_up(self):
		index_of_zero = np.where(self.state == 0)[0][0]
		# Boundary for up movement in grid - (0, 0), (0, 1), (0, 2)
		if index_of_zero in np.array([0, 1, 2]):
			return None
		else:
			# If not boundary, swap blank tile with down tile
			swap_value = self.state[index_of_zero - 3]
			new_state = self.state.copy()
			new_state[index_of_zero - 3] = 0
			new_state[index_of_zero] = swap_value
			return new_state

	def can_move_left(self):
		index_of_zero = np.where(self.state == 0)[0][0]
		# Boundary for left movement in grid - (0, 0), (1, 0), (2, 0)
		if index_of_zero in np.array([0, 3, 6]):
			return None
		else:
			# If not boundary, swap blank tile with right tile
			swap_value = self.state[index_of_zero - 1]
			new_state = self.state.copy()
			new_state[index_of_zero - 1] = 0
			new_state[index_of_zero] = swap_value
			return new_state

	def can_move_right(self):
		index_of_zero = np.where(self.state == 0)[0][0]
		# Boundary for right movement in grid - (0, 2), (1, 2), (2, 2)
		if index_of_zero in np.array([2, 5, 8]):
			return None
		else:
			# If not boundary, swap blank tile with left tile
			swap_value = self.state[index_of_zero + 1]
			new_state = self.state.copy()
			new_state[index_of_zero + 1] = 0
			new_state[index_of_zero] = swap_value
			return new_state

	def can_move_down(self):
		index_of_zero = np.where(self.state == 0)[0][0]
		# Boundary for down movement in grid - (2, 0), (2, 1), (2, 2)
		if index_of_zero in np.array([6, 7, 8]):
			return None
		else:
			# If not boundary, swap blank tile with up tile
			swap_value = self.state[index_of_zero + 3]
			new_state = self.state.copy()
			new_state[index_of_zero + 3] = 0
			new_state[index_of_zero] = swap_value
			return new_state

	def h_misplaced_cost(self, new_state, goal_state):
		''' Function for counting misplaced tiles '''
		misplaced_cost = 0
		for i in range(9):
			# if digit not in the grid position as that in the goal state, increase cost
			if new_state[i] != goal_state[i]:
				misplaced_cost += 1
		return misplaced_cost

	def h_manhattan_cost(self, new_state, goal_state):
		''' Function for calculating Manhattan distance '''
		current_node = new_state
		manhattan_cost = 0
		goal_state_map = {}
		for i in range(9):
			goal_state_map[goal_state[i]] = i

		for i in range(9):
			# count digit's distance from goal grid position
			manhattan_cost += abs(int(i/3) - int(goal_state_map[current_node[i]]/3)) + abs(i%3 - goal_state_map[current_node[i]]%3)
		return manhattan_cost

	def determine_h_cost(self, new_state, goal_state, heuristic):
		''' Helper function to determine which heuristic to use after taking input from user '''
		if heuristic == "1":
			return self.h_misplaced_cost(new_state, goal_state)
		else:
			return self.h_manhattan_cost(new_state, goal_state)

	def a_star(self, goal_state, heuristic):
		''' A* algorithm node explorer function '''
		# Priority queues for storing step costs
		priority_queue = [(self, 0)]
		g_cost_queue = [(0,0)]

		# Set that maintains visited node information
		visited = set([])

		# Variables for counting explored nodes and explanded nodes
		global explored_nodes
		global expanded_nodes
		explored_nodes = 0
		expanded_nodes = 0

		# Until queue is empty
		while priority_queue:
			# maintain priority queue
			priority_queue = sorted(priority_queue, key=lambda x: x[1])
			g_cost_queue = sorted(g_cost_queue, key=lambda x: x[1])

			# pop current node from queue
			current_node = priority_queue.pop(0)[0]
			current_g_cost = g_cost_queue.pop(0)[0]
			# add current node to visited node set
			visited.add(tuple(current_node.state.reshape(1,9)[0]))
			# increment explored node count
			explored_nodes += 1

			# If goal state has been reached, print the path taken to reach goal state
			if np.array_equal(current_node.state,goal_state):
				current_node.print_path()
				return True
			else:
				''' If goal state has not been reached, expand nodes for next move '''
				if current_node.can_move_up() is not None:
					new_state = current_node.can_move_up()
					if tuple(new_state.reshape(1,9)[0]) not in visited:
						g_cost = current_g_cost + 1
						h_cost = self.determine_h_cost(new_state, goal_state, heuristic)
						f_cost = g_cost + h_cost
						current_node.child_moves["up"] = Node(new_state, "up", g_cost, h_cost, current_node)
						priority_queue.append((current_node.child_moves["up"], f_cost))
						expanded_nodes += 1
						g_cost_queue.append((g_cost, f_cost))

				if current_node.can_move_left() is not None:
					new_state = current_node.can_move_left()
					if tuple(new_state.reshape(1,9)[0]) not in visited:
						g_cost = current_g_cost + 1
						h_cost = self.determine_h_cost(new_state, goal_state, heuristic)
						f_cost = g_cost + h_cost
						current_node.child_moves["left"] = Node(new_state, "left", g_cost, h_cost, current_node)
						priority_queue.append((current_node.child_moves["left"], f_cost))
						expanded_nodes += 1
						g_cost_queue.append((g_cost, f_cost))

				if current_node.can_move_right() is not None:
					new_state = current_node.can_move_right()
					if tuple(new_state.reshape(1,9)[0]) not in visited:
						g_cost = current_g_cost + 1
						h_cost = self.determine_h_cost(new_state, goal_state, heuristic)
						f_cost = g_cost + h_cost
						current_node.child_moves["right"] = Node(new_state, "right", g_cost, h_cost, current_node)
						priority_queue.append((current_node.child_moves["right"], f_cost))
						expanded_nodes += 1
						g_cost_queue.append((g_cost, f_cost))

				if current_node.can_move_down() is not None:
					new_state = current_node.can_move_down()
					if tuple(new_state.reshape(1,9)[0]) not in visited:
						g_cost = current_g_cost + 1
						h_cost = self.determine_h_cost(new_state, goal_state, heuristic)
						f_cost = g_cost + h_cost
						current_node.child_moves["down"] = Node(new_state, "down", g_cost, h_cost, current_node)
						priority_queue.append((current_node.child_moves["down"], f_cost))
						expanded_nodes += 1
						g_cost_queue.append((g_cost, f_cost))

	def print_path(self):
		''' Helper function to print the final path when the solution has been obtained '''
		trace = {"state": [self.state], "action": [self.action], "g_cost": [self.g_cost], "h_cost": [self.h_cost]}

		while self.parent:
			self = self.parent
			trace["state"].append(self.state)
			trace["action"].append(self.action)
			trace["g_cost"].append(self.g_cost)
			trace["h_cost"].append(self.h_cost)

		step_counter = 0
		while trace["state"]:
			print("Step: ", step_counter)
			print(trace["state"].pop())
			print("Action: ", trace["action"].pop())
			print("Path cost: ", trace["g_cost"].pop())
			print("Heuristic cost: ", trace["h_cost"].pop())
			print()
			step_counter += 1
		
		print("Explored nodes: ", explored_nodes)
		print("Expanded nodes: ", expanded_nodes)
		print()

def main():
	# Take input from user for initial and goal states
	initial_state = get_initial_state()
	goal_state = get_goal_state()
	# Take input for which heuristic to use
	heuristic = input("Which heuristic function would you like to use? Type '1' for misplaced tiles and '2' for Manhattan distance. If nothing is entered, the algorithm will use the Manhattan distance as a heuristic.\n")
	# Dummy node for calling function from Node class for initializing heuristic
	dummy_node = Node(goal_state, None, 0, 0, None)
	init_h_cost = dummy_node.determine_h_cost(initial_state, goal_state, heuristic)
	# Generating root node for tree initialization
	root_node = Node(initial_state, None, 0, init_h_cost, None)
	root_node.a_star(goal_state, heuristic)

main()