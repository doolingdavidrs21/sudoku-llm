import pandas as pd
import plotly.express as px
#import altair as alt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import math
import plotly.io as pio
import altair as alt
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import numpy as np
from random import shuffle
import copy
from dotenv import load_dotenv
import os
import ast
import openai
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # in memory

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA


class SudokuGenerator9X9:
	"""generates and solves Sudoku puzzles using a backtracking algorithm"""
	def __init__(self,grid=None):
		self.counter = 0
		#path is for the matplotlib animation
		self.path = []
		#if a grid/puzzle is passed in, make a copy and solve it
		if grid:
			if len(grid[0]) == 9 and len(grid) == 9:
				self.grid = grid
				self.original = copy.deepcopy(grid)
				self.solve_input_sudoku()
			else:
				print("input needs to be a 9x9 matrix")
		else:
			#if no puzzle is passed, generate one
			self.grid = [[0 for i in range(9)] for j in range(9)]
			self.generate_puzzle()
			self.original = copy.deepcopy(self.grid)
		
		
	def solve_input_sudoku(self):
		"""solves a puzzle"""
		self.generate_solution(self.grid)
		return

	def generate_puzzle(self):
		"""generates a new puzzle and solves it"""
		self.generate_solution(self.grid)
		#self.print_grid('full solution')
		self.remove_numbers_from_grid()
	#	self.print_grid('with removed numbers')
		return

	def print_grid(self, grid_name=None):
		if grid_name:
			print(grid_name)
		for row in self.grid:
			print(row)
		return

	def test_sudoku(self,grid):
		"""tests each square to make sure it is a valid puzzle"""
		for row in range(9):
			for col in range(9):
				num = grid[row][col]
				#remove number from grid to test if it's valid
				grid[row][col] = 0
				if not self.valid_location(grid,row,col,num):
					return False
				else:
					#put number back in grid
					grid[row][col] = num
		return True

	def num_used_in_row(self,grid,row,number):
		"""returns True if the number has been used in that row"""
		if number in grid[row]:
			return True
		return False

	def num_used_in_column(self,grid,col,number):
		"""returns True if the number has been used in that column"""
		for i in range(9):
			if grid[i][col] == number:
				return True
		return False

	def num_used_in_subgrid(self,grid,row,col,number):
		"""returns True if the number has been used in that subgrid/box"""
		sub_row = (row // 3) * 3
		sub_col = (col // 3)  * 3
		for i in range(sub_row, (sub_row + 3)): 
			for j in range(sub_col, (sub_col + 3)): 
				if grid[i][j] == number: 
					return True
		return False

	def valid_location(self,grid,row,col,number):
		"""return False if the number has been used in the row, column or subgrid"""
		if self.num_used_in_row(grid, row,number):
			return False
		elif self.num_used_in_column(grid,col,number):
			return False
		elif self.num_used_in_subgrid(grid,row,col,number):
			return False
		return True

	def find_empty_square(self,grid):
		"""return the next empty square coordinates in the grid"""
		for i in range(9):
			for j in range(9):
				if grid[i][j] == 0:
					return (i,j)
		return

	def solve_puzzle(self, grid):
		"""solve the sudoku puzzle with backtracking"""
		for i in range(0,81):
			row=i//9
			col=i%9
			#find next empty cell
			if grid[row][col]==0:
				for number in range(1,10):
					#check that the number hasn't been used in the row/col/subgrid
					if self.valid_location(grid,row,col,number):
						grid[row][col]=number
						if not self.find_empty_square(grid):
							self.counter+=1
							break
						else:
							if self.solve_puzzle(grid):
								return True
				break
		grid[row][col]=0  
		return False

	def generate_solution(self, grid):
		"""generates a full solution with backtracking"""
		number_list = [1,2,3,4,5,6,7,8,9]
		for i in range(0,81):
			row=i//9
			col=i%9
			#find next empty cell
			if grid[row][col]==0:
				shuffle(number_list)      
				for number in number_list:
					if self.valid_location(grid,row,col,number):
						self.path.append((number,row,col))
						grid[row][col]=number
						if not self.find_empty_square(grid):
							return True
						else:
							if self.generate_solution(grid):
								#if the grid is full
								return True
				break
		grid[row][col]=0  
		return False

	def get_non_empty_squares(self,grid):
		"""returns a shuffled list of non-empty squares in the puzzle"""
		non_empty_squares = []
		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i][j] != 0:
					non_empty_squares.append((i,j))
		shuffle(non_empty_squares)
		return non_empty_squares

	def remove_numbers_from_grid(self):
		"""remove numbers from the grid to create the puzzle"""
		#get all non-empty squares from the grid
		non_empty_squares = self.get_non_empty_squares(self.grid)
		non_empty_squares_count = len(non_empty_squares)
		rounds = 3
		while rounds > 0 and non_empty_squares_count >= 17:
			#there should be at least 17 clues
			row,col = non_empty_squares.pop()
			non_empty_squares_count -= 1
			#might need to put the square value back if there is more than one solution
			removed_square = self.grid[row][col]
			self.grid[row][col]=0
			#make a copy of the grid to solve
			grid_copy = copy.deepcopy(self.grid)
			#initialize solutions counter to zero
			self.counter=0      
			self.solve_puzzle(grid_copy)   
			#if there is more than one solution, put the last removed cell back into the grid
			if self.counter!=1:
				self.grid[row][col]=removed_square
				non_empty_squares_count += 1
				rounds -=1
		return

class SudokuGenerator4X4:
	"""generates and solves Sudoku puzzles using a backtracking algorithm"""
	def __init__(self,grid=None):
		self.counter = 0
		#path is for the matplotlib animation
		self.path = []
		#if a grid/puzzle is passed in, make a copy and solve it
		if grid:
			if len(grid[0]) == 4 and len(grid) == 4:
				self.grid = grid
				self.original = copy.deepcopy(grid)
				self.solve_input_sudoku()
			else:
				print("input needs to be a 4x4 matrix")
		else:
			#if no puzzle is passed, generate one
			self.grid = [[0 for i in range(4)] for j in range(4)]
			self.generate_puzzle()
			self.original = copy.deepcopy(self.grid)
		
		
	def solve_input_sudoku(self):
		"""solves a puzzle"""
		self.generate_solution(self.grid)
		return

	def generate_puzzle(self):
		"""generates a new puzzle and solves it"""
		self.generate_solution(self.grid)
		#self.print_grid('full solution')
		self.remove_numbers_from_grid()
		#self.print_grid('with removed numbers')
		return

	def print_grid(self, grid_name=None):
		if grid_name:
			print(grid_name)
		for row in self.grid:
			print(row)
		return

	def test_sudoku(self,grid):
		"""tests each square to make sure it is a valid puzzle"""
		for row in range(4):
			for col in range(4):
				num = grid[row][col]
				#remove number from grid to test if it's valid
				grid[row][col] = 0
				if not self.valid_location(grid,row,col,num):
					return False
				else:
					#put number back in grid
					grid[row][col] = num
		return True

	def num_used_in_row(self,grid,row,number):
		"""returns True if the number has been used in that row"""
		if number in grid[row]:
			return True
		return False

	def num_used_in_column(self,grid,col,number):
		"""returns True if the number has been used in that column"""
		for i in range(4):
			if grid[i][col] == number:
				return True
		return False

	def num_used_in_subgrid(self,grid,row,col,number):
		"""returns True if the number has been used in that subgrid/box"""
		sub_row = (row // 2) * 2
		sub_col = (col // 2)  * 2
		for i in range(sub_row, (sub_row + 2)): 
			for j in range(sub_col, (sub_col + 2)): 
				if grid[i][j] == number: 
					return True
		return False

	def valid_location(self,grid,row,col,number):
		"""return False if the number has been used in the row, column or subgrid"""
		if self.num_used_in_row(grid, row,number):
			return False
		elif self.num_used_in_column(grid,col,number):
			return False
		elif self.num_used_in_subgrid(grid,row,col,number):
			return False
		return True

	def find_empty_square(self,grid):
		"""return the next empty square coordinates in the grid"""
		for i in range(4):
			for j in range(4):
				if grid[i][j] == 0:
					return (i,j)
		return

	def solve_puzzle(self, grid):
		"""solve the sudoku puzzle with backtracking"""
		for i in range(0,16):
			row=i//4
			col=i%4
			#find next empty cell
			if grid[row][col]==0:
				for number in range(1,5):
					#check that the number hasn't been used in the row/col/subgrid
					if self.valid_location(grid,row,col,number):
						grid[row][col]=number
						if not self.find_empty_square(grid):
							self.counter+=1
							break
						else:
							if self.solve_puzzle(grid):
								return True
				break
		grid[row][col]=0  
		return False

	def generate_solution(self, grid):
		"""generates a full solution with backtracking"""
		number_list = [1,2,3,4]
		for i in range(0,16):
			row=i//4
			col=i%4
			#find next empty cell
			if grid[row][col]==0:
				shuffle(number_list)      
				for number in number_list:
					if self.valid_location(grid,row,col,number):
						self.path.append((number,row,col))
						grid[row][col]=number
						if not self.find_empty_square(grid):
							return True
						else:
							if self.generate_solution(grid):
								#if the grid is full
								return True
				break
		grid[row][col]=0  
		return False

	def get_non_empty_squares(self,grid):
		"""returns a shuffled list of non-empty squares in the puzzle"""
		non_empty_squares = []
		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i][j] != 0:
					non_empty_squares.append((i,j))
		shuffle(non_empty_squares)
		return non_empty_squares

	def remove_numbers_from_grid(self):
		"""remove numbers from the grid to create the puzzle"""
		#get all non-empty squares from the grid
        # 
		non_empty_squares = self.get_non_empty_squares(self.grid)
		non_empty_squares_count = len(non_empty_squares)
		rounds = 3
		while rounds > 0 and non_empty_squares_count >= 4:
			#there should be at least 17 clues
			row,col = non_empty_squares.pop()
			non_empty_squares_count -= 1
			#might need to put the square value back if there is more than one solution
			removed_square = self.grid[row][col]
			self.grid[row][col]=0
			#make a copy of the grid to solve
			grid_copy = copy.deepcopy(self.grid)
			#initialize solutions counter to zero
			self.counter=0      
			self.solve_puzzle(grid_copy)   
			#if there is more than one solution, put the last removed cell back into the grid
			if self.counter!=1:
				self.grid[row][col]=removed_square
				non_empty_squares_count += 1
				rounds -=1
		return


def make_4X4_puzzles(i:int = 10):
    """
    Function that returns a dictionary
    with integeger keys 0 through i.
    The values are a tuple of two lists of lists.
    the first is the 4X4 puzzle unsolved grid,
    the second is the unique solution.
    """
    puzzle4X4_dict = dict()
    for j in range(i):
        new_puzzle = SudokuGenerator4X4()
        dfpuzzle4X4 = pd.DataFrame.from_records(new_puzzle.grid)
        new_puzzle_sol = SudokuGenerator4X4(grid=dfpuzzle4X4.to_numpy().tolist())
        puzzle4X4_dict[j] = (new_puzzle.grid, new_puzzle_sol.grid)
    return puzzle4X4_dict


def make_9X9_puzzles(i:int = 10):
    """
    Function that returns a dictionary
    with integeger keys 0 through i.
    The values are a tuple of two lists of lists.
    the first is the 4X4 puzzle unsolved grid,
    the second is the unique solution.
    """
    puzzle9X9_dict = dict()
    for j in range(i):
        new_puzzle = SudokuGenerator9X9()
        dfpuzzle9x9 = pd.DataFrame.from_records(new_puzzle.grid)
        new_puzzle_sol = SudokuGenerator9X9(grid=dfpuzzle9X9.to_numpy().tolist())
        puzzle9X9_dict[j] = (new_puzzle.grid, new_puzzle_sol.grid)
    return puzzle9X9_dict




os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

